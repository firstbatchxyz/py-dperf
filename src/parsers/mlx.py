import mlx.nn as nn
import mlx.core as mx

from math import ceil
from dataclasses import dataclass
from typing import Any

from src.parsers.meta import LayerMeta

""" Estimate the FLOP count of all 'mlx_lm' models at a decoder level
    NOTE: Small OPs like RoPE and norms default to 0 FLOPs
    NOTE: FMA defaults to 2 FLOPs """

block_names = ["TransformerBlock", "DecoderLayer"]

# Add the quantization metadata to final byte count
def __quantized_bytes(n, d_bits, group_size, scale_bytes, zero_bytes):
    scaled_bits = n*d_bits
    code_bytes = ceil(scaled_bits/8)
    groups = (n + group_size-1) // group_size
    meta_bytes = groups * (scale_bytes + zero_bytes) 
    return code_bytes + meta_bytes 
    

def _profile_model(
    m: nn.Module,
    config: Any,
    B: int = 1,
    L: int = 4096,
    a_bits=16,
    w_bits=16,
    group_size=32,
    debug=0
):
    if not hasattr(m, "layers"):
        raise RuntimeError("Unable to profile a model without a '.layers' attribute.")

    decoder_idx = 1
    layers = []

    # Quantization hard-coded scale and zero bytes
    scale_bytes = 2
    zero_bytes = 0

    # Append a symbolic prefill layer to account for these FLOPs
    prefill = LayerMeta()
    prefill.name = "prefill"
    prefill.layer = None
    prefill.flops = 0
    prefill.kv_cache_r = 0
    prefill.kv_cache_w = 0
    layers.append(prefill)

    if debug >= 1:
        print(f"FMA: 2 FLOPs")
        #print(f"Quantization: {config.quantization.bits}")
        print(f"Parsing model {config.model_type}:")
        print(f"Quantization: bits={w_bits}, group_size={group_size}")
        print(f"    hidden_size={config.hidden_size},\n    vocab_size={config.vocab_size},\n"
              f"    num_hidden_layers={config.num_hidden_layers}")

    for l in m.layers:
        lm = LayerMeta()
        lm.layer = l
        lm.name = f"decoder_{decoder_idx}"
        if any(x in l.__class__.__name__ for x in ["TransformerBlock", "DecoderLayer"]):
            lm.input_bytes = (B * L * config.hidden_size * a_bits) / 8
            lm.output_bytes = (B * L * config.hidden_size * a_bits) / 8 
            if debug >= 1:
              print(f"\nParsing [decoder.{decoder_idx}]:")
            for name, obj in l.named_modules():
                if name == "post_attention_layernorm" or name == "input_layernorm":
                    pass

                elif name == "mlp":

                    # MoE
                    if (
                        hasattr(config, "n_routed_experts")
                        and hasattr(config, "first_k_dense_replace")
                        and hasattr(config, "moe_layer_freq")
                        and config.n_routed_experts is not None
                        and decoder_idx > config.first_k_dense_replace
                        and decoder_idx % config.moe_layer_freq == 0
                    ):
                        has_router_gate = False
                        gate_f, smlp_f, smlp_b, se_f, se_b = 0, 0, 0, 0, 0
                        for key, leaf in l[name].named_modules():
                            if key == "gate":
                                gate_f += ( 2 * B * config.hidden_size * config.n_routed_experts)
                                lm.flops += gate_f
                                has_router_gate = True

                            elif key == "switch_mlp":
                                DS = config.hidden_size * config.moe_intermediate_size
                                num_proj_smlp = 2
                                for key2, proj in leaf.named_modules():
                                    if  key2 == "gate_proj": 
                                        smlp_f += ( 2 * B * config.num_experts_per_tok * DS)
                                        num_proj_smlp = 3
                                    elif key2 in ["up_proj", "down_proj"]:
                                        smlp_f += ( 2 * B * config.num_experts_per_tok * DS)
                                    elif key2 == "activations":
                                        smlp_f += ( B * config.num_experts_per_tok * config.config.moe_intermediate_size)

                                # Add the quantization group overhead 
                                if w_bits < 16 and group_size is not None:
                                    bits = config.n_routed_experts*num_proj_smlp*config.hidden_size * config.moe_intermediate_size
                                    smlp_b = __quantized_bytes(bits, w_bits, group_size, scale_bytes, zero_bytes)
                                else: 
                                    smlp_b = ceil((config.n_routed_experts*num_proj_smlp*config.hidden_size
                                               * config.moe_intermediate_size * w_bits) / 8) 
                                lm.weight_bytes += smlp_b 
                                lm.flops += smlp_f

                            elif key == "shared_experts":
                                num_proj_se = 2
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        num_proj_se = 3
                                    if key2 in ["gate_proj", "up_proj", "down_proj"]:
                                        se_f += 2*B*config.hidden_size*config.n_shared_experts*config.moe_intermediate_size

                                if w_bits < 16 and group_size is not None:
                                    bits = config.n_shared_experts*num_proj_se*config.hidden_size * config.moe_intermediate_size 
                                    se_b = __quantized_bytes(bits, w_bits, group_size, scale_bytes, zero_bytes)
                                else: 
                                    se_b = ( config.n_shared_experts*num_proj_se*config.hidden_size
                                             * config.moe_intermediate_size * w_bits ) / 8
                                lm.weight_bytes += se_b
                                lm.flops += se_f

                        if debug >= 1:
                            print(f"\tMoE Layer: FLOPs={smlp_f+se_f+gate_f} ({num_proj_smlp}x{config.num_experts_per_tok}x"
                                  f"[{config.hidden_size}, {config.moe_intermediate_size}] + {num_proj_se}x"
                                  f"{config.n_shared_experts}x[{config.hidden_size}, {config.moe_intermediate_size}] + "
                                  f"{B}x[{config.hidden_size}, {config.n_routed_experts}]), b={smlp_b+se_b} @ {w_bits}bits" 
                                  if has_router_gate else f"), b={smlp_b+se_b} @ {w_bits}bits,", end="")
                            print(f" routed_experts={config.n_routed_experts} "
                                  f"with top-k={config.num_experts_per_tok}, ", end="")
                            print(f" shared_experts={config.n_shared_experts}")

                    # MLP
                    else:
                        num_proj = 2
                        proj_bytes = 0 
                        for key, leaf in l[name].named_modules():
                            if key == "gate_proj":
                                num_proj = 3
                            if key in ["gate_proj", "up_proj", "down_proj"]:
                                lm.flops += ( 2 * B * config.hidden_size * config.intermediate_size)
                                n = config.hidden_size * config.intermediate_size
                                if w_bits < 16 and group_size is not None:
                                    proj_bytes += __quantized_bytes(n, w_bits, group_size, scale_bytes, zero_bytes)
                                else: 
                                    proj_bytes += ceil((n*w_bits) / 8) 
                        lm.weight_bytes += proj_bytes


                        if debug >= 1:
                            print(f"\tMLP Layer: FLOPs={num_proj*2*B*config.hidden_size*config.intermediate_size},"
                                  f"  b={proj_bytes}"
                                  f"( {num_proj} x [{config.hidden_size}, {config.intermediate_size}] @ {w_bits}),"
                                  f"  b_i={B*config.hidden_size}([{B}, 1, {config.hidden_size}])")

                # NOTE: We only compute projection bits then correct in the case of quantization
                elif name == "self_attn":

                    is_gqa = False
                    is_mla = False

                    # Grouped Query Attention
                    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads != config.num_attention_heads:
                        is_gqa = True

                    # Low rank / Multi-head Latent Attention
                    if (all( hasattr(config, k) for k in [ "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"]) and
                        all( getattr(config, k) is not None for k in [ "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"])):
                        is_mla = True

                    if is_mla:
                        # Deepseek_v2,v3, Kimi_v1 and minicpm
                        if any(hasattr(config, k) for k in ["kv_lora_rank", "v_head_dim"]):

                            # Q projections, flops and bytes  
                            q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
                            q_a_proj = 2 * B * config.hidden_size * config.q_lora_rank
                            q_b_proj = 2 * B * config.num_attention_heads * q_head_dim * config.q_lora_rank
                            q_a_proj_n = config.q_lora_rank * config.hidden_size 
                            q_b_proj_n = config.num_attention_heads * q_head_dim * config.q_lora_rank

                            # KV projections
                            if is_gqa:
                                pass # TODO
                            else:
                                out_features = config.kv_lora_rank + config.qk_rope_head_dim
                                kv_a_proj_with_mqa = (2*B*(config.kv_lora_rank+config.qk_rope_head_dim)
                                                      * config.hidden_size)
                                kv_b_proj = ( 2*B* config.kv_lora_rank*config.num_attention_heads 
                                              *(config.qk_nope_head_dim+config.v_head_dim))
                                kv_a_proj_n = out_features * config.hidden_size 
                                kv_b_proj_n = out_features * config.kv_lora_rank

                            # O projection
                            o_proj = ( 2 * B * config.num_attention_heads * config.v_head_dim * config.hidden_size)
                            o_proj_n = config.hidden_size * config.num_attention_heads * config.v_head_dim


                            # KV Cache I/O
                            lm.kv_cache_w = ( B * (config.kv_lora_rank + config.kv_lora_rank) * a_bits) / 8 
                            lm.kv_cache_r = ( B * L * (config.kv_lora_rank + config.kv_lora_rank) * a_bits) / 8

                            # Totals
                            attn = 4 * B * L * config.q_lora_rank
                            lm.flops += ( q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj + attn)

                            if w_bits < 16 and group_size is not None:
                                q_a_proj_bytes  = __quantized_bytes(q_a_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                                q_b_proj_bytes  = __quantized_bytes(q_b_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                                kv_a_proj_bytes = __quantized_bytes(kv_a_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                                kv_b_proj_bytes = __quantized_bytes(kv_b_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                                o_proj_bytes    = __quantized_bytes(o_proj_n, w_bits, group_size, group_size, scale_bytes, zero_bytes)
                                attn_bytes = q_a_proj_bytes + q_b_proj_bytes + kv_a_proj_bytes + kv_b_proj_bytes + o_proj_bytes
                            else: 
                                q_a_proj_byts  = ceil((q_a_proj_n * w_bits) / 8)
                                q_b_proj_byts  = ceil((q_b_proj_n * w_bits) / 8)
                                kv_a_proj_byts = ceil((kv_a_proj_n * w_bits) / 8)
                                kv_b_proj_byts = ceil((kv_b_proj_n * w_bits) / 8)
                                o_proj_byts  = ceil((o_proj_n * w_bits) / 8)
                                attn_bytes = q_a_proj_bytes + q_b_proj_bytes + kv_a_proj_bytes + kv_b_proj_bytes + o_proj_bytes

                            lm.weight_bytes += attn_bytes 

                            if debug >= 1:
                                print(f"\tMulti-head Latent Attention Layer {"with Group Query Attention" if is_gqa else ""}:")
                                print(f"\t\tq_a_proj: [{config.hidden_size}, {config.q_lora_rank} ], FLOPs={q_a_proj}, b={q_a_proj_bytes}")
                                print(f"\t\tq_b_proj: [{config.num_attention_heads*q_head_dim}, {config.q_lora_rank}], "
                                      f"FLOPs={q_b_proj}, b={q_b_proj_bytes}")
                                print(f"\t\tkv_a_proj_with_mqa: [{config.hidden_size}, {config.kv_lora_rank + config.qk_rope_head_dim}], "
                                      f"FLOPs={kv_a_proj_with_mqa}, b={kv_a_proj_bytes}")
                                print(f"\t\tkv_b_proj: [{config.kv_lora_rank}, "
                                      f"{config.num_attention_heads*(config.qk_nope_head_dim+config.v_head_dim)}], "
                                      f"FLOPs={kv_b_proj}, b={kv_b_proj_bytes}")
                                print(f"\t\to_proj: [{config.num_attention_heads*config.v_head_dim}, {config.hidden_size}], "
                                      f"FLOPs={o_proj}, b={o_proj_bytes}")
                                print(f"\t\tFLOPs={q_a_proj + q_b_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj + attn} "
                                      f"b={attn_bytes}")

                            continue

                        # Low-rank Replace 
                        else:
                            psas

                    # Grouped Query Attention
                    elif is_gqa:
                        head_size = config.hidden_size / config.num_attention_heads
                        q_proj = 2 * B * config.hidden_size * config.hidden_size
                        q_proj_n = config.hidden_size * config.hidden_size

                        # TODO WRONG HEAD SIZE
                        k_proj = 2 * B * config.hidden_size * config.num_key_value_heads * head_size  
                        k_proj_n = config.hidden_size * (config.num_key_value_heads * head_size) 
                        v_proj = 2 * B * config.hidden_size * config.num_key_value_heads * head_size
                        v_proj_n = config.hidden_size * (config.num_key_value_heads * head_size)

                        o_proj = 2 * B * config.hidden_size * config.hidden_size
                        o_proj_n = config.hidden_size * config.hidden_size
                        attn = 4 * B * L * config.hidden_size

                        lm.flops += q_proj + k_proj + v_proj + o_proj + attn
                        lm.kv_cache_w = ( B * (2 * config.num_key_value_heads * head_size) * a_bits) / 8 
                        lm.kv_cache_r = ( B * L * (2 * config.num_key_value_heads * head_size) * a_bits) / 8

                        if w_bits < 16 and group_size is not None:
                            q_proj_bytes = __quantized_bytes(q_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                            k_proj_bytes = __quantized_bytes(k_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                            v_proj_bytes = __quantized_bytes(v_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                            o_proj_bytes = __quantized_bytes(o_proj_n, w_bits, group_size, scale_bytes, zero_bytes)
                            attn_bytes = q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes
                        else: 
                            q_proj_bytes = ceil((q_proj_n * w_bits) / 8)
                            k_proj_bytes = ceil((k_proj_n * w_bits) / 8)
                            v_proj_bytes = ceil((v_proj_n * w_bits) / 8)
                            o_proj_bytes = ceil((o_proj_n * w_bits) / 8)
                            attn_bytes = q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes

                        lm.weight_bytes += attn_bytes 

                        if debug >= 1:
                            print(f"\tGrouped Query Attention Layer: "
                                  f"FLOPs={8 * B * config.hidden_size * config.hidden_size + 4 * B * L * config.hidden_size}, "
                                  f"b={attn_bytes} "
                                  f"kv_cache_read={(B*L*(2*config.hidden_size) * a_bits) / 8}, "
                                  f"kv_cache_write={(B*(2*config.hidden_size) * a_bits) / 8}")
                            print(f"\t\tq_proj: FLOPs={q_proj}, b={q_proj_bytes}")
                            print(f"\t\tk_proj: FLOPs={k_proj}, b={k_proj_bytes}")
                            print(f"\t\tv_proj: FLOPs={v_proj}, b={v_proj_bytes}")
                            print(f"\t\to_proj: FLOPs={o_proj}, b={o_proj_bytes}")

                    # MHA
                    else:
                        lm.flops += ( 8 * B * config.hidden_size * config.hidden_size + 4 * B * L * config.hidden_size)
                        lm.kv_cache_w = (B * (2 * config.hidden_size) * a_bits ) / 8 
                        lm.kv_cache_r = (B * L * (2 * config.hidden_size) * a_bits) / 8 
                        n = 4 * config.hidden_size * config.hidden_size + 3 * config.hidden_size * config.intermediate_size

                        if w_bits < 16 and group_size is not None:
                            attn_bytes = __quantized_bytes(n, w_bits, group_size, scale_bytes, zero_bytes) 
                        else: 
                            attn_bytes = ceil((n*w_bits)/8) 

                        lm.weight_bytes = attn_bytes 

                        if debug >= 1:
                            print(f"\tAttention Layer: FLOPs={8 * B * config.hidden_size * config.hidden_size + 4 * B * L * config.hidden_size}, "
                                  f"b={attn_bytes} "
                                  f"kv_cache_read={(B * L * (2 * config.hidden_size) * a_bits) / 8}, "
                                  f"kv_cache_write={(B * (2 * config.hidden_size) * a_bits) / 8}")

            decoder_idx += 1
        layers.append(lm)
    return layers
