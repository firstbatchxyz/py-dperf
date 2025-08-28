from dataclasses import dataclass, field

@dataclass 
class CPUTopology:
    packages: int = 1
    cores: int = 0
    threads: int = 0

@dataclass
class CPUClock:
    base: float = 0.0 # MHz
    max: float = 0.0 # MHz

@dataclass
class CPUFeatures:
    AVX: bool = False
    FMA: bool = False
    BF16: bool = False
    SSE: bool = False

@dataclass
class CPUCache:
    l1d: int = 0
    l1i: int = 0
    l2: int = 0
    l3: int = 0

@dataclass 
class Stat:
    samples: int = 0
    min: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0 

@dataclass
class Benchmarks:
    flops_f64: float = 0.0
    flops_f32: float = 0.0
    flops_tf32: float = 0.0
    flops_fp16: float = 0.0
    flops_bf16: float = 0.0
    flops_u32: float = 0.0
    flops_u16: float = 0.0
    flops_u8: float = 0.0
    flops_i32: float = 0.0
    flops_i16: float = 0.0
    flops_i8: float = 0.0

@dataclass
class SystemMemory:
    can_swap: int = 0
    total: float = 0.0
    available: float = 0.0
    total_swap: float = 0.0
    available_swap: float = 0.0
    cpu_read_cold_bw: float = 0.0
    cpu_read_warm_bw: float = 0.0
    cpu_rw_cold_bw: float = 0.0
    cpu_rw_warm_bw: float = 0.0
    memcpy_delay: float = 0.0

@dataclass
class DiskInfo:
    read: float = 0.0
    write: float = 0.0
    random: float = 0.0

@dataclass
class CPUInfo:
    vendor: str = ""
    model: str = ""
    arch: str = "" 
    topology: CPUTopology = field(default_factory=CPUTopology)
    clock: CPUClock = field(default_factory=CPUClock)
    cache: CPUCache = field(default_factory=CPUCache)
    features: CPUFeatures = field(default_factory=CPUFeatures)
    benchmarks: Benchmarks = field(default_factory=Benchmarks)
    memcpy_hot: float = 0.0
    memcpy_cold: float = 0.0

@dataclass
class GPUMemory:
    name: str = ""
    free: float = 0
    total: float = 0
    read_bw: float = 0.0
    write_bw: float = 0.0
    read_write_bw: float = 0.0
    two_read_one_write_bw: float = 0.0
    vram_to_compute: float = 0.0
    unified_memory: bool = False

@dataclass
class GPUInfo:
    name: str = None 
    memory: GPUMemory = field(default_factory=GPUMemory)
    benchmarks: Benchmarks = field(default_factory=Benchmarks)

@dataclass
class DeviceInfo:
    os: str = None
    cpu: CPUInfo  = field(default_factory=CPUInfo)
    gpu: GPUInfo  = field(default_factory=GPUInfo)
    disk: DiskInfo = field(default_factory=DiskInfo)
    memory: SystemMemory = field(default_factory=SystemMemory)
    
