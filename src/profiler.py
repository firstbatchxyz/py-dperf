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
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

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
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median = stats.median(times)
    flop = 2.0 * N * M * K
    return flop / median * 1e-9


# MLX doesn't support multiprocessing for these ops, only separate streams with one op/stream
# Int datatype not supported on either device
def run_cpu_benchmarks(device_info):
    M = N = K = 2 << 8
    device_info.cpu.benchmarks.flops_f64 = _mlx_gemm_benchmark(
        mx.cpu, N, M, K, 3, 10, mx.float64
    )
    device_info.cpu.benchmarks.flops_f32 = _mlx_gemm_benchmark(
        mx.cpu, N, M, K, 3, 10, mx.float32
    )
    device_info.cpu.benchmarks.flops_fp16 = _mlx_gemm_benchmark(
        mx.cpu, N, M, K, 3, 10, mx.float16
    )
    device_info.cpu.benchmarks.flops_bf16 = _mlx_gemm_benchmark(
        mx.cpu, N, M, K, 3, 10, mx.bfloat16
    )
    # device_info.cpu.benchmarks.flops_u32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint32)
    # device_info.cpu.benchmarks.flops_u16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint16)
    # device_info.cpu.benchmarks.flops_u8  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint8)
    # device_info.cpu.benchmarks.flops_i32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int32)
    # device_info.cpu.benchmarks.flops_i16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int16)
    # device_info.cpu.benchmarks.flops_i8  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int8)


# consumer Nvidia GPUs also don't support f64
def run_gpu_benchmarks(device_info):
    M = N = K = 2 << 8
    # device_info.gpu.benchmarks.flops_f64  = _mlx_gemm_benchmark(mx.gpu, N,M,K, 3, 10, mx.float64)
    device_info.gpu.benchmarks.flops_f32 = _mlx_gemm_benchmark(
        mx.gpu, N, M, K, 3, 10, mx.float32
    )
    device_info.gpu.benchmarks.flops_fp16 = _mlx_gemm_benchmark(
        mx.gpu, N, M, K, 3, 10, mx.float16
    )
    device_info.gpu.benchmarks.flops_bf16 = _mlx_gemm_benchmark(
        mx.gpu, N, M, K, 3, 10, mx.bfloat16
    )
    # device_info.gpu.benchmarks.flops_u32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint32)
    # device_info.gpu.benchmarks.flops_u16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint16)
    # device_info.gpu.benchmarks.flops_u8  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.uint8)
    # device_info.gpu.benchmarks.flops_i32  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int32)
    # device_info.gpu.benchmarks.flops_i16  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int16)
    # device_info.gpu.benchmarks.flops_i8  = _mlx_gemm_benchmark(mx.cpu, N,M,K, 3, 10, mx.int8)


def bench_cpu_to_gpu_transfers(di):
    if _has_cupy:
        N = 2 << 8
        shape = N * N * N
        bytes_total = N * N * N * cp.dtype(cp.float32).itemsize

        def bench(fn, stream, warmup=3, iter=10):
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

        di.gpu.memory.read_bw = (bytes_total / bench(cpu_to_gpu, sec_cpu2gpu)) * 1e-9
        di.gpu.memory.write_bw = (bytes_total / bench(gpu_to_cpu, sec_gpu2cpu)) * 1e-9
        di.gpu.memory.read_write_bw = (
            2 * bytes_total / bench(read_write, sec_rw)
        ) * 1e-9


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

    # Transform in GB/s
    di.disk.write = ((M * M * M * 4) / w_time) * 1e-9
    di.disk.read = ((M * M * M * 4) / r_time) * 1e-9
    di.disk.random = (reads * BLOCK) / (rd_time * 1e9)


def bench(fn, warmup=3, iters=10):
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
    return stats.median(times)


# in GiB
def get_sysmem_info(device_info):
    import psutil, numpy as np

    sm = psutil.swap_memory()
    mx.set_default_device(mx.cpu)
    vm = psutil.virtual_memory()
    device_info.memory.total = vm.total * 1e-9
    device_info.memory.available = vm.available * 1e-9
    device_info.memory.total_swap = sm.total * 1e-9
    device_info.memory.available_swap = sm.free * 1e-9
    device_info.memory.can_swap = 1 if sm.total > 0 else 0

    M = 2 << 8
    A = mx.random.normal((M, M, M), dtype=mx.float32)
    B = np.random.randn(M, M, M)
    bytes_A = M * M * M * 4

    device_info.memory.cpu_read_cold_bw = (
        bytes_A / bench(lambda: mx.max(A), 0, 1)
    ) / 1e9

    t = 4
    parts = mx.split(A, t)
    streams = [mx.new_stream(mx.cpu) for _ in range(t)]

    def parallel_read_hot():
        return [mx.eval(mx.abs(p, stream=s)) for p, s in zip(parts, streams)]

    # device_info.memory.cpu_read_warm_bw = (bytes_A/bench(lambda: parallel_read_hot()))*1e-9
    device_info.memory.cpu_read_warm_bw = (
        bytes_A / bench(lambda: mx.abs(A), 5, 10)
    ) / 1e9
    device_info.memory.memcpy_delay = 1000 * bench(lambda: mx.eval(mx.array(B)))


# TODO: Maybee transfer this to the Metal package
def metal_get_memory_info(device_info):
    unified_mem = platform.machine() == "arm64"
    vm = psutil.virtual_memory()
    if unified_mem:
        device_info.gpu.name = "metal"
        device_info.gpu.unified = True
        device_info.gpu.total = vm.total
        device_info.gpu.free = vm.available
        # bench_gpu_transfer_times(device_info)
    # Skip the intel macbooks for now


# Get memory information
def cuda_get_memory_info(di):
    if _has_cupy:
        free, total = cp.cuda.runtime.memGetInfo()
        di.gpu.memory.total = total * 1e-9
        di.gpu.memory.free = free * 1e-9


def cuda_bench_mem_to_compute(di):
    pass


def metal_bench_mem_to_compute(di):
    M = 2 << 8
    s_gpu = mx.new_stream(mx.gpu)
    A = mx.random.normal((M, M, M), dtype=mx.float32, stream=s_gpu)
    B = mx.zeros_like(A, stream=s_gpu)
    zeros = mx.array(0.0, dtype=mx.float32)

    # Estimate the copy from RAM to compute units as cpy/2
    sec = bench(lambda: mx.add(A, zeros, stream=s_gpu))
    bw_cpy = (2 * M * M * M * 4) / sec
    bw_ram_read = bw_cpy / 2.0
    di.gpu.memory.vram_to_compute = bw_ram_read * 1e-9


# Solver-facing API


# Aggregate info on the current system
def profile() -> DeviceInfo:
    di = DeviceInfo()
    get_os(di)
    fill_cpu_info(di)
    run_cpu_benchmarks(di)
    run_gpu_benchmarks(di)
    if platform.system() == "Darwin":
        metal_bench_mem_to_compute(di)
        metal_get_memory_info(di)
        di.gpu.name = "metal"
    else:
        cuda_bench_mem_to_compute(di)
        cuda_get_memory_info(di)
        di.gpu.name = "cuda"
    bench_cpu_to_gpu_transfers(di)
    bench_disk_mainfs(di)
    get_sysmem_info(di)
    return di


@dataclass
class DeviceProfileInfo:
    b: List[int] = None
    b_i: List[int] = None
    b_o: List[int] = None
    d_avail: int = 0
    c_cpu: int = 0
    c_gpu: int = 0
    s_disk: float = 0.0

    # Values for a, b, c
    is_unified_mem: bool = False
    has_cuda: bool = False
    has_metal: bool = False
    os_type: str = ""
    f_q: List[int] = None
    s_cpu: Dict[str, float] = None
    s_gpu: Dict[str, float] = None
    t_kv_cpy_cpu: float = 0.0
    t_kv_cpy_gpu: float = 0.0
    tau_cpu: float = 0.0
    tau_gpu: float = 0.0
    t_ram_vram: float = 0.0
    t_vram_ram: float = 0.0
    t_comm: float = 0.0

    # Values for z, Z^gpu
    is_android: bool = False
    d_swapout: float = 0.0
    d_avail_cuda: float = 0.0
    d_avail_metal: float = 0.0

    def json(self):
        return json.dumps(asdict(self))

# Get device information in solver variable names 
def profile_device() -> DeviceProfileInfo:
    device_info = profile()
    ret = DeviceProfileInfo()
    ret.c_cpu = 0
    ret.c_gpu = 0
    ret.s_disk = device_info.disk.random
    ret.d_avail = device_info.memory.available
    ret.has_metal = True if device_info.gpu.name == "metal" else False
    ret.is_unified_mem = ret.has_metal # No intel macbooks support for now 
    ret.has_cuda = True if device_info.gpu.name == "cuda" else False
    ret.os_type = device_info.os
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
    ret.t_kv_cpy_cpu = device_info.memory.cpu_rw_cold_bw + device_info.memory.cpu_read_cold_bw
    if device_info.gpu.name == "cuda":
        ret.t_kv_cpy_gpu = device_info.gpu.memory.read_write_bw + device_info.gpu.memory.read_bw
    else:
        ret.t_kv_cpy_gpu = device_info.memory.cpu_rw_cold_bw + device_info.memory.cpu_read_cold_bw
    ret.tau_cpu = device_info.memory.cpu_read_warm_bw
    ret.tau_gpu = device_info.gpu.memory.vram_to_compute 
    if not ret.is_unified_mem:
        ret.t_ram_vram = device_info.gpu.memory.read_bw
        ret.t_vram_ram = device_info.gpu.memory.write_bw
    ret.is_android = False # no support
    ret.d_swapout = device_info.memory.total_swap
    ret.d_avail_cuda = device_info.gpu.memory.free if ret.has_cuda else 0.0
    ret.d_avail_metal = device_info.memory.available if ret.has_metal else 0.0
    return ret

# Estimate FLOPs for Model 
def profile_model(model: nn.Module, config, B: int=1, L: int=4096):
    model_info = _profile_model(model, config, B, L)
    ret = DeviceProfileInfo()
    ret.b =   [ x.weight_bytes for x in model_info ]
    ret.b_i = [ x.input_bytes  for x in model_info ]
    ret.b_o = [ x.output_bytes for x in model_info ]
    ret.f_q = [x.flops for x in model_info]
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
