from dataclasses import dataclass
from typing import Any


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
