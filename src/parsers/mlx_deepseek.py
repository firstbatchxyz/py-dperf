import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass
from typing import Any
from mlx_lm.models.deepseek_v2 import DeepseekV2DecoderLayer

# from src.model.base import IdentityBlock

"""
    Brute force the model profiling information for Deepseek V2.
    Parses the MLX Model object and returns an array of LayerMeta objects

    NOTE: Small OPs like RoPE and norms default to 0 (like prima.cpp)
    NOTE: FMA default to 2 FLOPs
"""


@dataclass
class LayerMeta:
    name: str = ""  # Name of the symbol
    submodules: Any = None  # Submodules decomposed into LayerMeta
    parent_layer: Any = None  # Parent Compount Layer
    layer: Any = None  # Original object
    flops: float = 0.0  # Estimated FLOPs to compute
    weight_bytes: int = 0  # Bytes of internal weight tensor
    input_bytes: int = 0  # Bytes of input tensor
    output_bytes: int = 0  # Bytes of output tensor
    kv_cache_t: int = 0  # Total tokens stored in KV Cache
    kv_cache_r: int = 0  # Bytes of KV Cache read
    kv_cache_w: int = 0  # Bytes of KV Cache written
    ram_vram_rw: int = 0  # Bytes of data transmitted between RAM <-> VRAM

    def __repr__(self):
        return (
            f"<LayerMeta {self.name}: "
            f"FLOPs={self.flops}, INPUT={self.input_bytes}, OUTPUT={self.output_bytes}, "
            f"WEIGHT={self.weight_bytes}, parent={self.parent_layer.__class__.__name__}"
        )


# Give batch size and sequence length
def profile_model(
    m: nn.Module,
    config: Any,
    B: int = 1,
    L: int = 4096,
    a_dtype=mx.float16,
    w_dtype=mx.float32,
):
    decoder_idx = 0  # MLP vs MoE
    layers = []
    for l in m.layers:
        if isinstance(l, DeepseekV2DecoderLayer):
            lm_decoder = LayerMeta()
            lm_decoder.layer = l
            lm_decoder.name = f"decode_{decoder_idx}"
            submodules = []
            for k in l.leaf_modules():
                if k == "":
                    continue
                elif k == "input_layernorm" or k == "post_attention_layernorm":
                    lm = LayerMeta()
                    lm.layer = l[k]
                    lm.name = k
                    lm.input_bytes = B * L * config.hidden_size * a_dtype.size
                    lm.output_bytes = B * L * config.hidden_size * a_dtype.size
                    submodules.append(lm)

                # MLA Block
                elif k == "self_attn":
                    for key, leaf in l[k].named_modules()[
                        ::-1
                    ]:  # Hack to reverse the symbol order
                        if key == "":
                            continue

                        lm = LayerMeta()
                        lm.parent_layer = l
                        lm.layer = leaf
                        lm.name = key

                        if hasattr(leaf, "weight"):
                            w_dtype = leaf.weight.dtype

                        # (B, L, D) -> (B, L, q_lora_rank)
                        if key == "q_a_proj":
                            lm.weight_bytes = (
                                config.q_lora_rank * config.hidden_size * w_dtype.size
                            )
                            lm.input_bytes = B * L * config.hidden_size * a_dtype.size
                            lm.output_bytes = B * L * config.q_lora_rank * a_dtype.size
                            lm.flops = (
                                2 * B * L * config.hidden_size * config.q_lora_rank
                            )
                            if config.attention_bias:
                                lm.flops = lm.flops + B * L * config.q_lora_rank

                        # (B, L, q_lora_rank) -> (B, L, num_heads*q_head_dim)
                        elif key == "q_b_proj":
                            q_head_dim = (
                                config.qk_nope_head_dim + config.qk_rope_head_dim
                            )
                            lm.weight_bytes = (
                                config.num_attention_heads
                                * q_head_dim
                                * config.q_lora_rank
                                * w_dtype.size
                            )
                            lm.input_bytes = B * L * config.q_lora_rank * a_dtype.size
                            lm.output_bytes = (
                                B
                                * L
                                * config.num_attention_heads
                                * q_head_dim
                                * a_dtype.size
                            )
                            lm.flops = (
                                2
                                * B
                                * L
                                * (config.num_attention_heads * q_head_dim)
                                * config.q_lora_rank
                            )

                        # (B, L, D) -> (B, L, kv_lora_rank+qk_rope_head_dim)
                        elif key == "kv_a_proj_with_mqa":
                            out_features = config.kv_lora_rank + config.qk_rope_head_dim
                            lm.weight_bytes = (
                                out_features * config.hidden_size * w_dtype.size
                            )
                            lm.input_bytes = B * L * config.hidden_size * a_dtype.size
                            lm.output_bytes = B * L * out_features * a_dtype.size
                            lm.flops = 2 * B * L * out_features * config.hidden_size
                            if config.attention_bias:
                                lm.flops = lm.flops + B * L * out_features

                        # (B, L, kv_lora_rank) ->
                        # (B, L, num_heads*(q_head_dim - v_head_dim + qk_rope_head_dim))
                        elif key == "kv_b_proj":
                            q_head_dim = (
                                config.qk_nope_head_dim * config.qk_rope_head_dim
                            )
                            out_features = config.num_attention_heads * (
                                config.qk_nope_head_dim + config.v_head_dim
                            )
                            lm.weight_bytes = (
                                out_features * config.kv_lora_rank * w_dtype.size
                            )
                            lm.input_bytes = B * L * config.kv_lora_rank * a_dtype.size
                            lm.output_bytes = B * L * out_features * a_dtype.size
                            lm.flops = 2 * B * L * config.kv_lora_rank * out_features

                        # (B, L, num_heads*v_head_dim) -> (B, L, D)
                        elif key == "o_proj":
                            dim = config.num_attention_heads * config.v_head_dim
                            lm.weight_bytes = config.hidden_size * dim * w_dtype.size
                            lm.input_bytes = B * L * out_features * a_dtype.size
                            lm.output_bytes = B * L * config.hidden_size * a_dtype.size
                            lm.flops = 2 * B * L * dim * config.hidden_size

                        elif key == "kv_a_layernorm":
                            lm.output_bytes = lm.input_bytes = (
                                B * L * config.kv_lora_rank * a_dtype.size
                            )
                            submodules.append(lm)
                            continue

                        elif key == "rope" or key == "q_a_layernorm":
                            continue  # No flops to count

                        submodules.append(lm)
                    decoder_idx = decoder_idx + 1

                elif k == "mlp":
                    # MoE is on if there are any routed experts
                    # Additional MLP layers added if there are any shared experts
                    if (
                        config.n_routed_experts is not None
                        and decoder_idx > config.first_k_dense_replace
                        and decoder_idx % config.moe_layer_freq == 0
                    ):

                        for key, leaf in l[k].named_modules()[::-1]:
                            if key == "":
                                continue

                            # MoEGate (B, L, D) -> (B*L, D) TODO: Bias?
                            if key == "gate":
                                lm = LayerMeta()
                                lm.parent_layer = l
                                lm.layer = leaf
                                lm.name = key
                                lm.weight_bytes = (
                                    config.hidden_size
                                    * config.n_routed_experts
                                    * a_dtype.size
                                )
                                lm.input_bytes = (
                                    B * L * config.hidden_size * a_dtype.size
                                )
                                lm.output_bytes = (
                                    B * L * config.n_routed_experts * a_dtype.size
                                )
                                lm.flops = (
                                    2
                                    * B
                                    * L
                                    * config.hidden_size
                                    * config.n_routed_experts
                                )
                                submodules.append(lm)

                            # SwitchGLU (B, L, D) -> (B, L, D)
                            elif key == "switch_mlp":
                                for key2, proj in leaf.named_modules()[::-1]:
                                    if key2 == "":
                                        continue
                                    lm = LayerMeta()
                                    lm.parent_layer = l
                                    lm.layer = proj
                                    lm.name = f"{key}.{key2}"
                                    DS = (
                                        config.hidden_size
                                        * config.moe_intermediate_size
                                    )

                                    # ((B*L)*num_experts_per_tok, D) ->
                                    # ((B*L)*num_experts_per_tok, moe_intermediate_size)
                                    if key2 == "gate_proj":
                                        lm.weight_bytes = DS * a_dtype.size
                                        lm.input_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.hidden_size
                                            * a_dtype.size
                                        )
                                        lm.output_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.moe_intermediate_size
                                            * a_dtype.size
                                        )
                                        lm.flops = (
                                            2
                                            * (B * L)
                                            * config.num_experts_per_tok
                                            * DS
                                        )

                                    # ((B*L)*num_experts_per_tok, D) ->
                                    # ((B*L)*num_experts_per_tok, moe_intermediate_size)
                                    elif key2 == "up_proj":
                                        lm.weight_bytes = DS * a_dtype.size
                                        lm.input_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.hidden_size
                                            * a_dtype.size
                                        )
                                        lm.output_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.moe_intermediate_size
                                            * a_dtype.size
                                        )
                                        lm.flops = (
                                            2
                                            * (B * L)
                                            * config.num_experts_per_tok
                                            * DS
                                        )

                                    # ((B*L)*num_experts_per_tok, moe_intermediate_size)
                                    elif key2 == "activation":
                                        lm.flops = (
                                            (B * L)
                                            * config.num_experts_per_tok
                                            * config.moe_intermediate_size
                                        )

                                    # ((B*L)*num_experts_per_tok, moe_intermediate_size) ->
                                    # ((B*L)*num_experts_per_tok, D)
                                    elif key2 == "down_proj":
                                        lm.weight_bytes = DS * a_dtype.size
                                        lm.input_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.moe_intermediate_size
                                            * a_dtype.size
                                        )
                                        lm.output_bytes = (
                                            B
                                            * L
                                            * config.num_experts_per_tok
                                            * config.hidden_size
                                            * a_dtype.size
                                        )
                                        lm.flops = (
                                            2
                                            * (B * L)
                                            * config.num_experts_per_tok
                                            * DS
                                        )

                                    submodules.append(lm)

                            elif key == "shared_experts":
                                lm = LayerMeta()
                                for key2, proj in leaf.named_modules()[::-1]:
                                    if key2 == "":
                                        continue
                                    lm = LayerMeta()
                                    lm.parent_layer = l
                                    lm.layer = leaf
                                    lm.name = f"{key}.{key2}"

                                    # ((B*L), D) -> ((B*L), n_shared_experts*moe_intermediate_size)
                                    if key2 == "gate_proj" or key2 == "up_proj":
                                        S_sh = (
                                            config.n_shared_experts
                                            * config.moe_intermediate_size
                                        )
                                        lm.weight_bytes = (
                                            S_sh * config.hidden_size * a_dtype.size
                                        )
                                        lm.input_bytes = (
                                            B * L * config.hidden_size * a_dtype.size
                                        )
                                        lm.output_bytes = (
                                            B
                                            * L
                                            * config.n_shared_experts
                                            * config.moe_intermediate_size
                                            * a_dtype.size
                                        )
                                        lm.flops = (
                                            2 * (B * L) * config.hidden_size * S_sh
                                        )

                                    # ((B*L), n_shared_experts*moe_intermediate_size) -> ((B*L), D)
                                    elif key2 == "down_proj":
                                        S_sh = (
                                            config.n_shared_experts
                                            * config.moe_intermediate_size
                                        )
                                        lm.weight_bytes = (
                                            S_sh * config.hidden_size * a_dtype.size
                                        )
                                        lm.input_bytes = (
                                            B
                                            * L
                                            * config.n_shared_experts
                                            * config.moe_intermediate_size
                                            * a_dtype.size
                                        )
                                        lm.output_bytes = (
                                            B * L * config.hidden_size * a_dtype.size
                                        )
                                        lm.flops = (
                                            2 * (B * L) * config.hidden_size * S_sh
                                        )

                                    submodules.append(lm)

                    # Compute MLP data
                    else:
                        for key, leaf in l[k].named_modules()[::-1]:
                            if key == "":
                                continue
                            lm = LayerMeta()
                            lm.parent_layer = l
                            lm.layer = leaf
                            lm.name = key

                            # ((B*L), D) -> ((B*L), intermediate_size)
                            if key == "gate_proj" or key == "up_proj":
                                lm.weight_bytes = (
                                    config.intermediate_size
                                    * config.hidden_size
                                    * a_dtype.size
                                )
                                lm.input_bytes = (
                                    B * L * config.hidden_size * a_dtype.size
                                )
                                lm.output_bytes = (
                                    B * L * config.intermediate_size * a_dtype.size
                                )
                                lm.flops = (
                                    2
                                    * (B * L)
                                    * config.hidden_size
                                    * config.intermediate_size
                                )

                            # ((B*L), intermediate_size) -> ((B*L), D)
                            elif key == "down_proj":
                                lm.weight_dtype = (
                                    config.intermediate_size
                                    * config.hidden_size
                                    * a_dtype.size
                                )
                                lm.input_bytes = (
                                    B * L * config.intermediate_size * a_dtype.size
                                )
                                lm.output_bytes = (
                                    B * L * config.hidden_size * a_dtype.size
                                )
                                lm.flops = (
                                    2
                                    * (B * L)
                                    * config.hidden_size
                                    * config.intermediate_size
                                )

                            submodules.append(lm)

            lm_decoder.submodules = [x for x in submodules if x is not None]
            lm_decoder.input_bytes = lm_decoder.submodules[0].input_bytes
            lm_decoder.output_bytes = lm_decoder.submodules[-1].output_bytes
            for x in lm_decoder.submodules:
                lm_decoder.flops += x.flops
                lm_decoder.weight_bytes += x.weight_bytes

            # Add extra work done by initial layer
            if decoder_idx == 1:
                # Embedding
                emlm = LayerMeta()
                emlm.name = "embedding"
                emlm.weight_bytes = (
                    config.vocab_size * config.hidden_size * w_dtype.size
                )
                emlm.output_bytes = B * L * config.hidden_size * a_dtype.size
                lm_decoder.submodules.insert(0, emlm)

            # decoder-scoped KV cache bytes
            lm_decoder.kv_cache_r = (
                B
                * (
                    config.num_key_value_heads
                    * L
                    * (config.qk_nope_head_dim + config.qk_rope_head_dim)
                    + config.num_attention_heads * L * config.v_head_dim
                )
                * a_dtype.size
            )
            lm_decoder.kv_cache_w = (
                B
                * (
                    config.num_key_value_heads
                    * (config.qk_nope_head_dim + config.qk_rope_head_dim)
                    + config.num_attention_heads * config.v_head_dim
                )
                * a_dtype.size
            )
            layers.append(lm_decoder)

        # elif isinstance(l, IdentityBlock):
        #    pass # TODO: Maybe add ad empty block

    # Add extra work done by the last decoder layer
    rmslm = LayerMeta()
    rmslm.name = "final_layernorm"
    rmslm.input_bytes = B * L * config.hidden_size * a_dtype.size
    rmslm.output_bytes = B * L * config.hidden_size * a_dtype.size
    layers[-1].submodules.append(rmslm)

    return layers
