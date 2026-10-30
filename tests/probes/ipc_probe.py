"""Two processes, CUDA IPC, as the engine's p2p exchange: A allocates a region on its GPU and publishes the IPC handle;
B (its own GPU first, A's visible second) opens it with lazy peer access, then copies a pattern into it with
cuMemcpyDtoD(Async) and reads A's pattern out of it. Both check the bytes. Times every copy.

    python ipc_probe.py            (spawns B itself)"""
import ctypes, os, subprocess, sys, time, mmap

cu = ctypes.CDLL("libcuda.so.1")
N = 256 << 20
SHM = "/dev/shm/ipc-probe"


def ck(rc, what):
    if rc:
        name = ctypes.c_char_p()
        cu.cuGetErrorName(rc, ctypes.byref(name))
        raise SystemExit(f"{what}: {rc} {name.value}")


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def ctx(ordinal=0):
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
    c = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), dev), "retain")
    ck(cu.cuCtxSetCurrent(c), "setcurrent")


def fill(ptr, byte):
    ck(cu.cuMemsetD8_v2(ctypes.c_uint64(ptr), ctypes.c_ubyte(byte), ctypes.c_size_t(N)), "memset")  # noqa
    ck(cu.cuCtxSynchronize(), "sync")


def check(ptr, byte, who):
    host = (ctypes.c_ubyte * N)()
    ck(cu.cuMemcpyDtoH_v2(host, ctypes.c_uint64(ptr), ctypes.c_size_t(N)), "dtoh")
    bad = sum(1 for i in range(0, N, 4096) if host[i] != byte)
    print(f"  {who}: {'OK' if bad == 0 else f'{bad} of {N // 4096} sampled bytes wrong'} (want {byte}, first {host[0]})", flush=True)


def main_a():
    ctx()
    p = ctypes.c_uint64()
    ck(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(2 * N)), "alloc")
    fill(p.value, 0xA0)                                   # A's half: 0xA0 ...
    print("A self-check:", flush=True); check(p.value, 0xA0, "A's own half after memset")
    ck(cu.cuMemsetD8_v2(ctypes.c_uint64(p.value + N), ctypes.c_ubyte(0), ctypes.c_size_t(N)), "memset0")
    h = Handle()
    ck(cu.cuIpcGetMemHandle(ctypes.byref(h), p), "ipcget")
    with open(SHM, "wb") as f:
        f.write(ctypes.string_at(ctypes.addressof(h), 64))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("PROBE_B_CARDS", "1,0")}
    b = subprocess.run([sys.executable, __file__, "B"], env=env)
    ck(cu.cuCtxSynchronize(), "sync")
    print("A checks what B wrote into its second half:", flush=True)
    check(p.value + N, 0xB1, "A's region, B's half")
    os.unlink(SHM)


def main_b():
    ctx(int(os.environ.get("PROBE_B_ORDINAL", "0")))
    raw = open(SHM, "rb").read()
    h = Handle(); ctypes.memmove(ctypes.addressof(h), raw, 64)
    peer = ctypes.c_uint64()
    cu.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), Handle, ctypes.c_uint]
    ck(cu.cuIpcOpenMemHandle_v2(ctypes.byref(peer), h, 1), "ipcopen")
    ordinal, kind = ctypes.c_int(-1), ctypes.c_uint(0)
    rc1 = cu.cuPointerGetAttribute(ctypes.byref(ordinal), 9, peer)   # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    rc2 = cu.cuPointerGetAttribute(ctypes.byref(kind), 2, peer)      # CU_POINTER_ATTRIBUTE_MEMORY_TYPE (2 = device)
    print(f"B: opened pointer {peer.value:#x}: device ordinal {ordinal.value} (rc {rc1}), memory type {kind.value} (rc {rc2})", flush=True)
    mine = ctypes.c_uint64()
    ck(cu.cuMemAlloc_v2(ctypes.byref(mine), ctypes.c_size_t(N)), "alloc")
    for label, fn in (("cuMemcpyDtoD_v2", lambda d, s: cu.cuMemcpyDtoD_v2(ctypes.c_uint64(d), ctypes.c_uint64(s), ctypes.c_size_t(N))),
                      ("cuMemcpyDtoDAsync_v2 (legacy stream)", lambda d, s: cu.cuMemcpyDtoDAsync_v2(ctypes.c_uint64(d), ctypes.c_uint64(s), ctypes.c_size_t(N), None))):
        print(f"B, {label}:", flush=True)
        fill(mine.value, 0xB1)
        check(mine.value, 0xB1, "B's own buffer after memset")
        t = time.perf_counter(); ck(fn(peer.value + N, mine.value), "push"); ck(cu.cuCtxSynchronize(), "sync")
        dt = time.perf_counter() - t
        print(f"  push {N >> 20} MB into A's region: {dt * 1e3:.1f} ms = {N / dt / 1e9:.1f} GB/s", flush=True)
        fill(mine.value, 0)
        t = time.perf_counter(); ck(fn(mine.value, peer.value), "pull"); ck(cu.cuCtxSynchronize(), "sync")
        dt = time.perf_counter() - t
        print(f"  pull {N >> 20} MB out of A's region: {dt * 1e3:.1f} ms = {N / dt / 1e9:.1f} GB/s", flush=True)
        check(mine.value, 0xA0, "B's copy of A's half")
    ck(cu.cuIpcCloseMemHandle(peer), "ipcclose")
    # control: B's own two buffers, a plain device-to-device copy
    other = ctypes.c_uint64(); ck(cu.cuMemAlloc_v2(ctypes.byref(other), ctypes.c_size_t(N)), "alloc2")
    fill(mine.value, 0x5C); fill(other.value, 0)
    t = time.perf_counter(); ck(cu.cuMemcpyDtoD_v2(other, mine, ctypes.c_size_t(N)), "local"); ck(cu.cuCtxSynchronize(), "sync")
    print(f"B local copy: {N / (time.perf_counter() - t) / 1e9:.1f} GB/s", flush=True); check(other.value, 0x5C, "B local copy")


if __name__ == "__main__":
    main_b() if sys.argv[1:] == ["B"] else main_a()
