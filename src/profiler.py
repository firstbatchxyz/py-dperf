import os
import sys
import uuid
import mmap
import json
import struct
import psutil
import pathlib
import platform
import random
import multiprocessing
import time, statistics as stats

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

DEBUG = 1

# Type alias for quantization performance metrics
QuantPerf = Dict[str, float]

from cpuinfo import get_cpu_info

from dataclasses import asdict
from src.datatypes import DeviceInfo
from src.parsers.mlx import _profile_model

# from src.utils.logger import logger

try:
    import cupy as cp
    import ctypes as C

    _has_cupy = True
except ImportError as e:
    print("Unable to load CuPy library for CUDA benchmarking.")
    cp = None
    _has_cupy = False


def get_os(device_info):
    device_info.os = platform.system()


def fill_cpu_info(di):
    info = get_cpu_info()
    di.cpu.model = info["brand_raw"]
    di.cpu.arch = ["arch_string_raw"]

    # cpuid instruction only on x86
    if info["arch_string_raw"] in ["x86_64", "amd64"]:
        di.cpu.clock.base = info["hz_actual"][0]
        di.cpu.clock.max = info["hz_advertised"][0]
        di.cpu.topology.cores = (
            info["count"] // 2 if "ht" in info["flags"] else info["count"]
        )
        di.cpu.topology.threads = info["count"]
        di.cpu.vendor = info["vendor_id_raw"]
        if info["arch_string_raw"] in ["x86_64", "amd64", "aarch64", "arm64"]:
            di.cpu.arch = info["arch_string_raw"]
        else:
            raise TypeError(f"Unsupported CPU architecture {info.arch_string.raw}")
        di.cpu.features.AVX = True if "avx" in info["flags"] else False
        di.cpu.features.AVX2 = True if "avx2" in info["flags"] else False
        di.cpu.features.FMA = True if "fma" in info["flags"] else False
        di.cpu.features.BF16 = True if "bf16" in info["flags"] else False
        di.cpu.features.SSE = True if "sse" in info["flags"] else False
        if di.cpu.arch == "aarch64" or di.cpu.arch == "arm64":
            di.cpu.features.NEON = True if "neon" in info["flags"] else False
        di.cpu.cache.l3 = info["l3_cache_size"] * 1e-6
        di.cpu.cache.l2 = info["l2_cache_size"] * 1e-6
        di.cpu.cache.l1d = info["l1_data_cache_size"] * 1e-6
        di.cpu.cache.l1i = info["l1_instruction_cache_size"] * 1e-6

    # Apple
    else:
        di.cpu.topology.cores = psutil.cpu_count(logical=False)
        di.cpu.topology.threads = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            di.cpu.clock.base = cpu_freq.current
            di.cpu.clock.max = cpu_freq.max
        uname = platform.uname()
        di.cpu.vendor = uname.processor or uname.system or ""
        di.cpu.model = uname.machine or ""


# Dense gemm benchmark
def _mlx_gemm_benchmark(
    device, N, M, K, warmup: int = 3, iters: int = 10, dtype=mx.float32
) -> float:
    try:
        mx.set_default_device(device)
        A = mx.random.normal((M, K), dtype=dtype)
        B = mx.random.normal((K, N), dtype=dtype)

        for _ in range(warmup):
            C = mx.matmul(A, B)
            mx.eval(C)
        mx.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            C = mx.matmul(A, B)
            mx.eval(C)
            mx.synchronize()
            times.append(time.perf_counter() - t0)

        median = stats.median(times)
        flop = 2.0 * N * M * K

        if DEBUG:
            mean = stats.mean(times) * 1000 
            std  = stats.stdev(times) * 1000
            p50  = stats.median(times) * 1000
            p95  = stats.quantiles(times, n=iters)[iters-5] * 1000
            p99  = stats.quantiles(times, n=iters)[iters-2] * 1000
            mean_gflop = stats.mean([ (flop/t)*1e-9 for t in times])

            print(f"gemm {N}x{M}@{K}x{N} ({dtype}, {device})")
            print(f"    {iters} runs [ms]: avg {mean:5.3f} ± {std:.3f}  "
                  f" p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}")
            print(f"    [GFLOP/s]: {mean_gflop:.3f}")
        return flop / median  # Return FLOPS, not GFLOPS
    except:
        return 0.0

# MLX doesn't support multiprocessing for these ops, only separate streams with one op/stream
# Int datatype not supported on either device
def run_cpu_benchmarks(device_info, n_embd: int):
    M = N = K = int(n_embd/8 if n_embd >= 4096 else 4096/8) # Smaller size on CPU
    device_info.cpu.benchmarks.flops_f64  = _mlx_gemm_benchmark( mx.cpu, N, M, K, 5, 100, mx.float64)
    device_info.cpu.benchmarks.flops_f32  = _mlx_gemm_benchmark( mx.cpu, N, M, K, 5, 100, mx.float32)
    device_info.cpu.benchmarks.flops_fp16 = _mlx_gemm_benchmark( mx.cpu, N, M, K, 5, 100, mx.float16)
    device_info.cpu.benchmarks.flops_bf16 = _mlx_gemm_benchmark( mx.cpu, N, M, K, 5, 100, mx.bfloat16)
    device_info.cpu.benchmarks.flops_u32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint32)
    device_info.cpu.benchmarks.flops_u16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint16)
    device_info.cpu.benchmarks.flops_u8   = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint8)
    device_info.cpu.benchmarks.flops_i32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int32)
    device_info.cpu.benchmarks.flops_i16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int16)
    device_info.cpu.benchmarks.flops_i8   = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int8)

# consumer Nvidia GPUs don't support f64
def run_gpu_benchmarks(device_info, n_embd: int):
    M = N = K = n_embd if n_embd >= 4096 else 4096
    device_info.gpu.benchmarks.flops_f64  = _mlx_gemm_benchmark(mx.gpu, N,M,K, 3, 10, mx.float64)
    device_info.gpu.benchmarks.flops_f32  = _mlx_gemm_benchmark( mx.gpu, N, M, K, 20, 60, mx.float32)
    device_info.gpu.benchmarks.flops_fp16 = _mlx_gemm_benchmark( mx.gpu, N, M, K, 20, 60, mx.float16)
    device_info.gpu.benchmarks.flops_bf16 = _mlx_gemm_benchmark( mx.gpu, N, M, K, 20, 60, mx.bfloat16)
    device_info.gpu.benchmarks.flops_u32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint32)
    device_info.gpu.benchmarks.flops_u16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint16)
    device_info.gpu.benchmarks.flops_u8   = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint8)
    device_info.gpu.benchmarks.flops_i32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int32)
    device_info.gpu.benchmarks.flops_i16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int16)
    device_info.gpu.benchmarks.flops_i8   = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int8)


def bench_cpu_to_gpu_transfers(di, n_embd):
    if _has_cupy:  # Benchmark VRAM <-> RAM through CUDA
        N = n_embd if n_embd >= 4096 else 4096
        bytes_total = N * N * cp.dtype(cp.float32).itemsize
        shape = N * N 

        def bench(fn, stream, name, warmup=3, iter=10):
            times = []
            for _ in range(warmup):
                fn()
            for _ in range(iter):
                t0 = cp.cuda.Event()
                t1 = cp.cuda.Event()
                with stream:
                    t0.record()
                    fn()
                    t1.record()
                t1.synchronize()
                times.append(cp.cuda.get_elapsed_time(t0, t1) / 1000.0)  # seconds

            if DEBUG:
                mean = stats.mean(times) * 1000 
                std  = stats.stdev(times) * 1000
                p50  = stats.median(times) * 1000
                p95  = stats.quantiles(times, n=iter)[iter-5] * 1000
                p99  = stats.quantiles(times, n=iter)[iter-2] * 1000

                print(f"{name}: {iter} runs [ms]: avg {mean:5.3f} ± {std:.3f}  "
                      f" p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}")

            return stats.median(times)

        d = cp.empty(shape, dtype=cp.float32)
        h_in = cp.cuda.alloc_pinned_memory(bytes_total)
        h_out = cp.cuda.alloc_pinned_memory(bytes_total)
        h_in_ptr, h_out_ptr = h_in.ptr, h_out.ptr

        sec_cpu2gpu = cp.cuda.Stream(non_blocking=True)
        sec_gpu2cpu = cp.cuda.Stream(non_blocking=True)
        sec_rw = cp.cuda.Stream(non_blocking=True)

        def cpu_to_gpu():
            with sec_cpu2gpu:
                cp.cuda.runtime.memcpyAsync(
                    d.data.ptr,
                    h_in_ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyHostToDevice,
                    sec_cpu2gpu.ptr,
                )

        def gpu_to_cpu():
            with sec_gpu2cpu:
                cp.cuda.runtime.memcpyAsync(
                    h_out_ptr,
                    d.data.ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    sec_gpu2cpu.ptr,
                )

        def read_write():
            with sec_cpu2gpu:
                cp.cuda.runtime.memcpyAsync(
                    d.data.ptr,
                    h_in_ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyHostToDevice,
                    sec_cpu2gpu.ptr,
                )
                
            with sec_gpu2cpu:
                cp.cuda.runtime.memcpyAsync(
                    h_out_ptr,
                    d.data.ptr,
                    bytes_total,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    sec_gpu2cpu.ptr,
                )

        di.gpu.memory.read_bw = bytes_total / bench(cpu_to_gpu, sec_cpu2gpu, "cpu_to_gpu")  # bytes/s
        di.gpu.memory.write_bw = bytes_total / bench(gpu_to_cpu, sec_gpu2cpu, "gpu_to_cpu")  # bytes/s
        di.gpu.memory.read_write_bw = ( 2 * bytes_total / bench(read_write, sec_rw, "read_write"))  # bytes/s


def bench_disk_mainfs(di, iter=10, reads=200):
    M = 2 << 8
    BLOCK = 1024  # Number of bytes accessed per read, 256 f32
    A = mx.random.normal((M, M, M), dtype=mx.float32)  # ~536MB

    # Force the commit to disk
    def fsync(fd):
        try:
            # if platform.system() == "Darwin":  # Stronger flush on OSX
            #    fcntl.fcntl(fd, fnctl.F_FULLFSYNC)
            # else:
            os.fsync(fd)
        finally:
            os.close(fd)

    # Write to disk
    times = []
    for _ in range(iter):
        path = pathlib.Path(__file__).parent.resolve() / f"__tmp_{uuid.uuid4().hex}.npy"
        t0 = time.perf_counter()
        mx.save(str(path), A)
        fd = os.open(path, os.O_RDWR)
        fsync(fd)
        times.append(time.perf_counter() - t0)
        os.remove(path)  # Remove file after every run
    w_time = sum(times) / iter

    # Random mmap blocked access
    rng = random.Random(80085)
    offsets = [rng.randrange(0, M * M * M // BLOCK) * BLOCK for _ in range(reads)]

    path = pathlib.Path(__file__).parent.resolve() / f"__tmp_{uuid.uuid4().hex}.npy"
    mx.save(str(path), A)

    with open(path, "rb") as fd:
        mm = mmap.mmap(fd.fileno(), M * M * M * 4, access=mmap.ACCESS_READ)
        total = 0
        t0 = time.perf_counter()
        for off in offsets:
            b = mm[off : off + BLOCK]
            total += b[0]  # touch data
        rd_time = time.perf_counter() - t0
        mm.close()

    # Full file read into memory
    times = []
    for _ in range(iter):
        t0 = time.perf_counter()
        obj = mx.load(str(path))
        mx.eval(obj)
        times.append(time.perf_counter() - t0)
    r_time = sum(times) / iter
    os.remove(path)

    # Transform to bytes/s
    di.disk.write = (M * M * M * 4) / w_time  # bytes/s
    di.disk.read = (M * M * M * 4) / r_time  # bytes/s
    di.disk.random = (reads * BLOCK) / rd_time  # bytes/s


def bench(fn, name="", warmup=3, iters=10):
    times = []
    for _ in range(warmup):
        mx.synchronize()
        mx.eval(fn())
        mx.synchronize()
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
        mx.synchronize()

    if DEBUG and len(times) > 2:
        mean = stats.mean(times) * 1000 
        std  = stats.stdev(times) * 1000
        p50  = stats.median(times) * 1000
        p95  = stats.quantiles(times, n=iters)[iters-5] * 1000
        p99  = stats.quantiles(times, n=iters)[iters-2] * 1000

        print(f"{name:10}: {iters} runs [ms]: avg {mean:5.3f} ± {std:.3f}  "
              f" p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}")

    return stats.median(times)


def get_sysmem_info(device_info, config):
    import psutil, numpy as np

    sm = psutil.swap_memory()
    mx.set_default_device(mx.cpu)
    vm = psutil.virtual_memory()
    device_info.memory.total = vm.total  # bytes
    device_info.memory.available = vm.available  # bytes
    device_info.memory.total_swap = sm.total  # bytes
    device_info.memory.available_swap = sm.free  # bytes
    device_info.memory.can_swap = True if sm.total > 0 else False 

    M = 2 << 8
    A = mx.random.normal((M, M, M), dtype=mx.float32)
    B = np.random.randn(M, M, M)
    bytes_A = M * M * M * 4

    device_info.memory.cpu_read_cold_bw = bytes_A / bench(
        lambda: mx.max(A), "cpy_read_cold_bw", 0, 1
    )  # bytes/s

    t = 4
    parts = mx.split(A, t)
    streams = [mx.new_stream(mx.cpu) for _ in range(t)]

    def parallel_read_hot():
        return [mx.eval(mx.abs(p, stream=s)) for p, s in zip(parts, streams)]

    # device_info.memory.cpu_read_warm_bw = bytes_A/bench(lambda: parallel_read_hot())  # bytes/s
    device_info.memory.cpu_read_warm_bw = bytes_A / bench(
        lambda: mx.abs(A), "cpu_read_warm_bw", 5, 10
    )  # bytes/s

    device_info.memory.cpu_write_cold_bw = bytes_A / bench(
        lambda: mx.full((M*M*M), 23.4, dtype=mx.float32), "cpu_write_cold_bw", 0, 1
    )

    device_info.memory.cpu_write_warm_bw = bytes_A / bench(
        lambda: mx.full((M*M*M), 351.23, dtype=mx.float32), "cpu_write_warm_bw", 5, 10
    )

    device_info.memory.memcpy_delay = 1000 * bench(lambda: mx.eval(mx.array(B)), "memcpy_delay")


# TODO: Maybee transfer this to the Metal package
def metal_get_memory_info(device_info):
    unified_mem = platform.machine() == "arm64"
    vm = psutil.virtual_memory()
    if unified_mem:
        device_info.gpu.name = "metal"
        device_info.gpu.memory.unified_memory = True
        device_info.gpu.memory.total = vm.total  # bytes
        device_info.gpu.memory.free = vm.available  # bytes
        # bench_gpu_transfer_times(device_info)
    # Skip the intel macbooks for now


# Get memory information
def cuda_get_memory_info(di):
    if _has_cupy:
        free, total = cp.cuda.runtime.memGetInfo()
        di.gpu.memory.total = total  # bytes
        di.gpu.memory.free = free  # bytes


def cuda_bench_mem_to_compute(di):
    pass


# Best aproximation, still short of ~100GB/s expected
def metal_bench_mem_to_compute(di):
    M = 512 
    s_gpu = mx.new_stream(mx.gpu)

    # Randomize to escape caching
    A = mx.random.normal((8*M, M, M), dtype=mx.float32, stream=s_gpu)
    idxs = list(range(8))

    # Estimate the copy from RAM to compute units 
    def mem_load():
        i = random.choice(idxs)
        #out = mx.add(A[i*M:(i+1)*M], 0.0, stream=s_gpu)
        out = mx.sum(A[i*M:(i+1)*M])
        mx.eval(out)

    sec = bench(mem_load, "vram_to_compute", 30, 100)
    bw_cpy = (2 * M * M * M * 4) / sec
    bw_ram_read = bw_cpy 
    di.gpu.memory.vram_to_compute = bw_ram_read  # bytes/s


# Solver-facing API


# Aggregate info on the current system
def profile(config) -> DeviceInfo:
    di = DeviceInfo()
    get_os(di)
    fill_cpu_info(di)
    run_cpu_benchmarks(di, config.hidden_size)
    run_gpu_benchmarks(di, config.hidden_size)
    if platform.system() == "Darwin":
        metal_bench_mem_to_compute(di)
        metal_get_memory_info(di)
        di.gpu.name = "metal"
    else:
        cuda_bench_mem_to_compute(di)
        cuda_get_memory_info(di)
        di.gpu.name = "cuda"
    bench_cpu_to_gpu_transfers(di, config.hidden_size)
    bench_disk_mainfs(di)
    get_sysmem_info(di, config)
    return di


@dataclass
class ModelProfileInfo:
    """
    Model-global constants (bytes, sizes, FLOPs) from profiler.
    """

    # Per-layer metrics (existing fields)
    b: List[int] = None  # bytes per layer (list)
    b_i: List[int] = None  # input bytes per layer (list)
    b_o: List[int] = None  # output bytes per layer (list)
    f_q: List[int] = None  # FLOPs per layer (list)

    # Model-level metrics (new fields)
    L: int = 0  # total layers
    hk: int = 0  # heads for keys
    ek: int = 0  # emb per head (k)
    hv: int = 0  # heads for values
    ev: int = 0  # emb per head (v)
    n_kv: int = 0  # tokens in KV cache
    e_embed: int = 0  # embedding size
    V: int = 0  # vocabulary size

    # FLOPs per layer for each quantization
    f_by_quant: QuantPerf = field(default_factory=dict)  # f_q (per "typical" layer)
    f_out_by_quant: QuantPerf = field(
        default_factory=dict
    )  # f_{q, out} (for output layer)
    Q: List[str] = field(
        default_factory=lambda: ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "F32"]
    )


@dataclass
class DeviceProfileInfo:
    """
    One device dm with measured/profiler data.
    Notation in comments matches the paper's symbols.
    """

    # --- required (no defaults) ---
    name: str = ""
    os_type: str = ""  # 'mac_no_metal' | 'mac_metal' | 'linux' | 'android'
    is_head: bool = (
        True  # I_{m=1}  (True for the head device that holds input/output layers on CPU)
    )
    is_unified_mem: bool = False  # I_UMA (Apple Silicon etc.)
    has_cuda: bool = False  # I_cuda
    has_metal: bool = False  # I_metal

    # Throughput tables (FLOPS) per quantization for CPU/GPU paths
    scpu: QuantPerf = None  # s^{cpu}_{m,q}
    T_cpu: float = 0.0  # T^{cpu}_m (register loading throughput, bytes/s)

    # KV-copy times (sec) for a fixed 2*(h_k e_k + h_v e_v)·n_kv byte payload
    t_kvcpy_cpu: float = 0.0  # t^{kv_cpy,cpu}_m
    t_kvcpy_gpu: float = 0.0  # t^{kv_cpy,gpu}_m

    # Host<->GPU staging + inter-device comm (sec)
    t_ram2vram: float = 0.0  # t^{ram->vram}_m
    t_vram2ram: float = 0.0  # t^{vram->ram}_m
    t_comm: float = 0.0  # t^{comm}_m

    # Disk read throughput (bytes/s)
    s_disk: float = 0.0  # s^{disk}_m

    # Available memories / working sets (bytes)
    d_avail_ram: int = 0  # d^{avail}_m (RAM)

    # --- optional (come after required) ---
    sgpu_cuda: QuantPerf = None  # s^{gpu}_{m,q} for CUDA
    sgpu_metal: QuantPerf = None  # s^{gpu}_{m,q} for Metal
    T_cuda: float = None  # T^{gpu}_m for CUDA (bytes/s)
    T_metal: float = None  # T^{gpu}_m for Metal (bytes/s)
    d_avail_cuda: int = None  # d^{avail}_{m,cuda} (VRAM)
    d_avail_metal: int = None  # d^{avail}_{m,metal} (Metal working set)

    # --- small buffers and swap caps (bytes) ---
    c_cpu: int = 0  # c^{cpu} (CPU compute buffer)
    c_gpu: int = 0  # c^{gpu} (GPU compute buffer)

    # Android swap capacity (only used if os_type == "android")
    d_bytes_can_swap: int = 0  # potential bytes we allow swapping
    d_swap_avail: int = 0  # actually available swap bytes

    def json(self):
        return json.dumps(asdict(self))


# Get device information in solver variable names
def profile_device(config) -> DeviceProfileInfo:
    device_info = profile(config)
    ret = DeviceProfileInfo()

    # Set device name (hostname or identifier)
    ret.name = platform.node() or "device"

    # Determine OS type with metal/no-metal distinction
    ret.has_metal = True if device_info.gpu.name == "metal" else False
    ret.has_cuda = True if device_info.gpu.name == "cuda" else False
    ret.is_unified_mem = ret.has_metal  # Apple Silicon has unified memory

    # Set OS type based on platform and GPU availability
    if platform.system() == "Darwin":
        if ret.has_metal:
            ret.os_type = "mac_metal"
        else:
            ret.os_type = "mac_no_metal"
    elif platform.system() == "Linux":
        # Check if Android
        try:
            with open("/proc/version", "r") as f:
                if "android" in f.read().lower():
                    ret.os_type = "android"
                else:
                    ret.os_type = "linux"
        except:
            ret.os_type = "linux"
    else:
        ret.os_type = "linux"  # Default fallback

    # Set is_head to True by default (single device scenario)
    ret.is_head = True

    # CPU throughput tables (FLOPS)
    ret.scpu = {
        "f64": device_info.cpu.benchmarks.flops_f64,
        "f32": device_info.cpu.benchmarks.flops_f32,
        "fp16": device_info.cpu.benchmarks.flops_fp16,
        "bf16": device_info.cpu.benchmarks.flops_bf16,
    }

    # CPU register loading throughput (bytes/s) - use warm bandwidth
    ret.T_cpu = device_info.memory.cpu_read_warm_bw  # Already in bytes/s

    # GPU throughput tables (FLOPS) - separate for CUDA and Metal
    if ret.has_cuda:
        ret.sgpu_cuda = {
            "f32": device_info.gpu.benchmarks.flops_f32,
            "fp16": device_info.gpu.benchmarks.flops_fp16,
            "bf16": device_info.gpu.benchmarks.flops_bf16,
        }
        # CUDA memory throughput (bytes/s)
        ret.T_cuda = device_info.gpu.memory.vram_to_compute  # Already in bytes/s
    elif ret.has_metal:
        ret.sgpu_metal = {
            "f32": device_info.gpu.benchmarks.flops_f32,
            "fp16": device_info.gpu.benchmarks.flops_fp16,
            "bf16": device_info.gpu.benchmarks.flops_bf16,
        }
        # Metal memory throughput (bytes/s)
        ret.T_metal = device_info.gpu.memory.vram_to_compute  # Already in bytes/s

    # KV-copy times (sec) - time for a standard KV operation
    # Using a standard 1MB payload for timing calculation
    kv_payload_size = 1024 * 1024  # 1MB standard payload
    #kv_payload_size = 2 * config.hidden_size*mx.float16.itemsize # 1 KV copy

    #cpu_bw = device_info.memory.cpu_rw_cold_bw + device_info.memory.cpu_read_cold_bw
    cpu_bw = device_info.memory.cpu_read_cold_bw*2 + device_info.memory.cpu_write_cold_bw
    if cpu_bw > 0:
        ret.t_kvcpy_cpu = kv_payload_size / cpu_bw  # seconds for 1MB KV copy

    if device_info.gpu.name == "cuda":
        gpu_bw = device_info.gpu.memory.read_write_bw + device_info.gpu.memory.read_bw
        if gpu_bw > 0:
            ret.t_kvcpy_gpu = kv_payload_size / gpu_bw  # seconds for 1MB KV copy
    elif ret.has_metal:
        if cpu_bw > 0:
            ret.t_kvcpy_gpu = (
                kv_payload_size / cpu_bw
            )  # seconds for 1MB KV copy (unified memory)

    # Host<->GPU staging times (sec) - time for 1MB transfer
    transfer_size = 1024 * 1024  # 1MB standard transfer
    if not ret.is_unified_mem:
        if device_info.gpu.memory.read_bw > 0:
            ret.t_ram2vram = transfer_size / device_info.gpu.memory.read_bw  # seconds
        if device_info.gpu.memory.write_bw > 0:
            ret.t_vram2ram = transfer_size / device_info.gpu.memory.write_bw  # seconds

    # Inter-device communication time (0 for single device)
    ret.t_comm = 0.0

    # Disk read throughput (bytes/s)
    ret.s_disk = device_info.disk.random  # Already in bytes/s

    # Available memories (already in bytes)
    ret.d_avail_ram = int(device_info.memory.available)

    if ret.has_cuda:
        ret.d_avail_cuda = int(device_info.gpu.memory.free)
    elif ret.has_metal:
        ret.d_avail_metal = int(device_info.memory.available)  # Unified memory

    # Small buffers (bytes) - set to 0 for now
    ret.c_cpu = 0
    ret.c_gpu = 0

    # Swap capacity (already in bytes)
    ret.d_bytes_can_swap = int(device_info.memory.total_swap)
    ret.d_swap_avail = int(device_info.memory.available_swap)

    return ret


# Estimate FLOPs for Model
def profile_model(
    model: nn.Module, config, B: int = 1, L: int = 4096, config_dict: Dict = None
):
    model_info = _profile_model(model, config, B, L)
    ret = ModelProfileInfo()

    # Per-layer metrics (existing)
    ret.b = [x.weight_bytes for x in model_info]
    ret.b_i = [x.input_bytes for x in model_info]
    ret.b_o = [x.output_bytes for x in model_info]
    ret.f_q = [x.flops for x in model_info]

    # Use config_dict if available for more complete access, otherwise fall back to config object
    cfg = config_dict if config_dict else {}

    # Model-level metrics from config
    ret.L = cfg.get(
        "num_hidden_layers", getattr(config, "num_hidden_layers", len(model_info) - 1)
    )
    ret.e_embed = cfg.get("hidden_size", getattr(config, "hidden_size", 0))
    ret.V = cfg.get("vocab_size", getattr(config, "vocab_size", 0))

    # Attention head configuration
    num_attention_heads = cfg.get(
        "num_attention_heads", getattr(config, "num_attention_heads", 0)
    )
    ret.hk = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )
    ret.hv = cfg.get(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", num_attention_heads),
    )

    # Calculate head dimension
    head_dim = cfg.get("head_dim", getattr(config, "head_dim", 0))
    if head_dim == 0 and ret.e_embed > 0 and num_attention_heads > 0:
        head_dim = ret.e_embed // num_attention_heads
    ret.ek = head_dim
    ret.ev = head_dim

    # KV cache tokens (using max position embeddings as proxy)
    ret.n_kv = cfg.get(
        "max_position_embeddings", getattr(config, "max_position_embeddings", L)
    )

    # Get quantization info from config if available
    quant_info = cfg.get("quantization", cfg.get("quantization_config", {}))
    bits = quant_info.get("bits", 32) if isinstance(quant_info, dict) else 32

    # Calculate quantization FLOPs based on actual quantization bits
    if ret.f_q and len(ret.f_q) > 1:
        typical_layer_flops = ret.f_q[1] if len(ret.f_q) > 1 else ret.f_q[0]
        output_layer_flops = ret.f_q[-1] if ret.f_q else 0

        # More accurate estimates based on bit width
        quant_scales = {
            "Q4_K": 4.0 / 32.0,
            "Q5_K": 5.0 / 32.0,
            "Q6_K": 6.0 / 32.0,
            "Q8_0": 8.0 / 32.0,
            "F16": 16.0 / 32.0,
            "F32": 1.0,
        }

        # Adjust scales if we know the actual quantization
        if bits == 4:
            base_scale = 4.0 / 32.0
        elif bits == 8:
            base_scale = 8.0 / 32.0
        elif bits == 16:
            base_scale = 16.0 / 32.0
        else:
            base_scale = 1.0

        for quant, scale in quant_scales.items():
            ret.f_by_quant[quant] = typical_layer_flops * scale
            ret.f_out_by_quant[quant] = output_layer_flops * scale

    return ret


# Get solver variables including estimated model flops/bytes
def get_solver_vars(model: nn.Module, config, B: int = 1, L: int = 4096):
    device_info = profile_device()
    model_info = profile_model(model, config, B, L)

    ret = DeviceProfileInfo()

    ret.b = [x.weight_bytes for x in model_info]
    ret.b_i = [x.input_bytes for x in model_info]
    ret.b_o = [x.output_bytes for x in model_info]
    ret.d_avail = device_info.memory.available
    ret.c_cpu = 0
    ret.c_gpu = 0
    ret.s_disk = device_info.disk.random

    ret.has_metal = True if device_info.gpu.name == "metal" else False
    ret.is_unified_mem = ret.has_metal  # No intel macbooks support for now
    ret.has_cuda = True if device_info.gpu.name == "cuda" else False
    ret.os_type = device_info.os
    ret.f_q = [x.flops for x in model_info]
    ret.s_cpu = {
        "f64": device_info.cpu.benchmarks.flops_f64,
        "f32": device_info.cpu.benchmarks.flops_f32,
        "fp16": device_info.cpu.benchmarks.flops_fp16,
        "bf16": device_info.cpu.benchmarks.flops_bf16,
    }
    ret.s_gpu = {
        "f32": device_info.gpu.benchmarks.flops_f32,
        "fp16": device_info.gpu.benchmarks.flops_fp16,
        "bf16": device_info.gpu.benchmarks.flops_bf16,
    }
    # Estimate 2 reads + 1 write
    ret.t_kv_cpy_cpu = (
        device_info.memory.cpu_rw_cold_bw + device_info.memory.cpu_read_cold_bw
    )
    ret.t_kv_cpy_gpu = device_info.gpu.memory.two_read_one_write_bw
    ret.tau_cpu = device_info.memory.cpu_read_warm_bw
    ret.tau_gpu = device_info.gpu.memory.vram_to_compute
    if not ret.is_unified_mem:
        ret.t_ram_vram = device_info.gpu.memory.read_bw
        ret.t_vram_ram = device_info.gpu.memory.write_bw
    ret.t_comm = 0.0
    ret.is_android = False  # no support
    ret.d_swapout = device_info.memory.total_swap
    ret.d_avail_cuda = device_info.gpu.memory.free if ret.has_cuda else 0.0
    ret.d_avail_metal = device_info.gpu.memory.free if ret.has_metal else 0.0

    return ret
