import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass
from typing import Any

from src.parsers.meta import LayerMeta

""" Estimate the FLOP count of all 'mlx_lm' models at a decoder level
    NOTE: Small OPs like RoPE and norms default to 0 FLOPs
    NOTE: FMA defaults to 2 FLOPs """

block_names = ["TransformerBlock", "DecoderLayer"]


def _profile_model(
    m: nn.Module,
    config: Any,
    B: int = 1,
    L: int = 4096,
    a_dtype=mx.float16,
    w_dtype=mx.float32,
):
    if not hasattr(m, "layers"):
        raise RuntimeError("Unable to profile a model without a '.layers' attribute.")

    decoder_idx = 0
    layers = []

    # Append a symbolic prefill layer to account for these FLOPs
    prefill = LayerMeta()
    prefill.name = "prefill"
    prefill.layer = None
    prefill.flops = 0
    prefill.kv_cache_r = 0
    prefill.kv_cache_w = 0
    layers.append(prefill)

    for l in m.layers:
        lm = LayerMeta()
        lm.layer = l
        lm.name = f"decoder_{decoder_idx}"
        if any(x in l.__class__.__name__ for x in ["TransformerBlock", "DecoderLayer"]):
            lm.input_bytes = B * L * config.hidden_size * a_dtype.size
            lm.output_bytes = B * L * config.hidden_size * a_dtype.size
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

                        for key, leaf in l[name].named_modules():
                            if key == "gate":
                                lm.flops += (
                                    2 * B * config.hidden_size * config.n_routed_experts
                                )
                            elif key == "switch_mlp":
                                lm.weight_bytes += config.n_routed_experts * (
                                    2
                                    * config.hidden_size
                                    * config.moe_intermediate_size
                                )
                                DS = config.hidden_size * config.moe_intermediate_size
                                for key2, proj in leaf.named_modules():
                                    if (
                                        key2 == "gate_proj"
                                    ):  # Only count gate when present
                                        lm.weight_bytes += config.n_routed_experts * (
                                            config.hidden_size
                                            * config.moe_intermediate_size
                                        )
                                    if key2 in ["gate_proj", "up_proj", "down_proj"]:
                                        lm.flops += (
                                            2 * B * config.num_experts_per_tok * DS
                                        )
                                    elif key2 == "activations":
                                        lm.flops += (
                                            B
                                            * config.num_experts_per_tok
                                            * config.config.moe_intermediate_size
                                        )

                            elif key == "shared_experts":
                                lm.weight_bytes += config.n_shared_experts * (
                                    2
                                    * config.hidden_size
                                    * config.moe_intermediate_size
                                )
                                S_sh = (
                                    config.n_shared_experts
                                    * config.moe_intermediate_size
                                )
                                for key2, proj in leaf.named_modules():
                                    if key2 == "gate_proj":
                                        lm.weight_bytes += config.n_shared_experts * (
                                            config.hidden_size
                                            * config.moe_intermediate_size
                                        )
                                    if key2 in ["gate_proj", "up_proj", "down_proj"]:
                                        lm.flops += 2 * B**config.hidden_size * S_sh
                    # MLP
                    else:
                        lm.weight_bytes += (
                            2 * config.hidden_size * config.intermediate_size
                        )
                        for key, leaf in l[name].named_modules():
                            if key == "gate_proj":
                                lm.weight_bytes += (
                                    config.hidden_size * config.intermediate_size
                                )
                            if key in ["gate_proj", "up_proj", "down_proj"]:
                                lm.flops += (
                                    2
                                    * B
                                    * config.hidden_size
                                    * config.intermediate_size
                                )

                elif name == "self_attn":
                    if (
                        hasattr(config, "num_key_value_heads")
                        and config.num_key_value_heads != config.num_attention_heads
                    ):  # GQA

                        head_size = config.hidden_size / config.num_attention_heads
                        q_proj = 2 * B * config.hidden_size * config.hidden_size
                        k_proj = (
                            2
                            * B
                            * config.hidden_size
                            * config.num_key_value_heads
                            * head_size
                        )  # TODO WRONG HEAD SIZE
                        v_proj = (
                            2
                            * B
                            * config.hidden_size
                            * config.num_key_value_heads
                            * head_size
                        )
                        o_proj = 2 * B * config.hidden_size * config.hidden_size
                        attn = 4 * B * L * config.hidden_size

                        # Weight size in bytes
                        q_bytes = config.hidden_size * config.hidden_size * w_dtype.size
                        k_bytes = (
                            config.hidden_size
                            * (config.num_key_value_heads * head_size)
                            * w_dtype.size
                        )
                        v_bytes = (
                            config.hidden_size
                            * (config.num_key_value_heads * head_size)
                            * w_dtype.size
                        )
                        o_bytes = (
                            (config.num_key_value_heads * head_size)
                            * config.hidden_size
                            * w_dtype.size
                        )

                        # Low rank
                        if all(
                            hasattr(config, k)
                            for k in [
                                "q_lora_rank",
                                "qk_nope_head_dim",
                                "qk_rope_head_dim",
                            ]
                        ):
                            if hasattr(l, "o_proj"):
                                if (
                                    l.o_proj.weight.shape[1] == config.hidden_size
                                ):  # Low-rank replace
                                    pass

                                # MLA, Deepseek_v2,v3, Kimi_v1 and minicpm
                                else:
                                    q_head_dim = (
                                        config.qk_nope_head_dim
                                        + config.qk_rope_head_dim
                                    )
                                    q_a_proj = (
                                        2 * B * config.hidden_size * config.q_lora_rank
                                    )
                                    q_b_proj = (
                                        2
                                        * B
                                        * config.num_attention_heads
                                        * q_head_dim
                                        * config.q_lora_rank
                                    )
                                    kv_a_proj_with_mqa = (
                                        2
                                        * B
                                        * L
                                        * (
                                            config.kv_lora_rank
                                            + config.qk_rope_head_dim
                                        )
                                        * config.hidden_size
                                    )
                                    kv_b_proj = (
                                        2
                                        * B
                                        * config.kv_lora_rank
                                        * config.num_attention_heads
                                        * (config.qk_nope_head_dim + config.v_head_dim)
                                    )
                                    o_proj = (
                                        2
                                        * B
                                        * config.num_attention_heads
                                        * config.v_head_dim
                                        * config.hidden_size
                                    )
                                    attn = 4 * B * L * config.q_lora_rank
                                    lm.flops += (
                                        q_a_proj
                                        + q_b_proj
                                        + kv_a_proj_with_mqa
                                        + kv_b_proj
                                        + o_proj
                                        + attn
                                    )

                                    lm.kv_cache_w = (
                                        B
                                        * (config.kv_lora_rank + config.kv_lora_rank)
                                        * a_dtype.size
                                    )
                                    lm.kv_cache_r = (
                                        B
                                        * L
                                        * (config.kv_lora_rank + config.kv_lora_rank)
                                        * a_dtype.size
                                    )

                                    # TODO Check with deepseek impl
                                    q_a_proj_bytes = (
                                        config.q_lora_rank
                                        * config.hidden_size
                                        * w_dtype.size
                                    )
                                    q_b_proj_bytes = (
                                        config.num_attention_heads
                                        * q_head_dim
                                        * config.q_lora_rank
                                        * w_dtype.size
                                    )
                                    kv_a_proj_bytes = (
                                        out_features * config.hidden_size * w_dtype.size
                                    )
                                    kv_b_proj_bytes = (
                                        out_features
                                        * config.kv_lora_rank
                                        * w_dtype.size
                                    )
                                    o_proj_bytes = (
                                        config.hidden_size
                                        * config.num_attention_heads
                                        * config.v_head_dim
                                        * w_dtype.size
                                    )
                                    lm.weight_bytes += (
                                        q_a_proj_bytes
                                        + q_b_proj_bytes
                                        + kv_a_proj_bytes
                                        + kv_b_proj_bytes
                                        + o_proj_bytes
                                    )

                                    continue

                        # Standard GQA
                        lm.flops += q_proj + k_proj + v_proj + o_proj + attn
                        lm.kv_cache_w = (
                            B
                            * (2 * config.num_key_value_heads * head_size)
                            * a_dtype.size
                        )
                        lm.kv_cache_r = (
                            B
                            * L
                            * (2 * config.num_key_value_heads * head_size)
                            * a_dtype.size
                        )

                    # Vanilla
                    else:
                        lm.flops += (
                            8 * B * config.hidden_size * config.hidden_size
                            + 4 * B * L * config.hidden_size
                        )
                        lm.kv_cache_w = B * (2 * config.hidden_size) * a_dtype.size
                        lm.kv_cache_r = B * L * (2 * config.hidden_size) * a_dtype.size
                        lm.weight_bytes = (
                            4 * config.hidden_size * config.hidden_size
                            + 3 * config.hidden_size * config.intermediate_size
                        )

            decoder_idx += 1
        layers.append(lm)
    return layers
