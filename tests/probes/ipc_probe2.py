"""The layout without cross-GPU IPC: process B (sees its GPU first, A's second) allocates X on A's GPU and Y on its own,
copies between them in-process (peer access enabled), and exports X to A by IPC (same GPU for A). Checks bytes, times.
    CUDA_VISIBLE_DEVICES=0,1 python ipc_probe2.py"""
import ctypes, os, subprocess, sys, time

cu = ctypes.CDLL("libcuda.so.1")
N = 256 << 20
SHM = "/dev/shm/ipc-probe2"


def ck(rc, what):
    if rc:
        name = ctypes.c_char_p(); cu.cuGetErrorName(rc, ctypes.byref(name))
        raise SystemExit(f"{what}: {rc} {name.value}")


class Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def context(ordinal):
    dev = ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev), ordinal), "dev")
    c = ctypes.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), dev), "retain")
    return c


def alloc(c):
    ck(cu.cuCtxSetCurrent(c), "set"); p = ctypes.c_uint64(); ck(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(N)), "alloc"); return p


def fill(c, p, b):
    ck(cu.cuCtxSetCurrent(c), "set"); ck(cu.cuMemsetD8_v2(p, ctypes.c_ubyte(b), ctypes.c_size_t(N)), "memset"); ck(cu.cuCtxSynchronize(), "sync")


def check(c, p, b, who):
    ck(cu.cuCtxSetCurrent(c), "set")
    host = (ctypes.c_ubyte * N)(); ck(cu.cuMemcpyDtoH_v2(host, p, ctypes.c_size_t(N)), "dtoh")
    bad = sum(1 for i in range(0, N, 4096) if host[i] != b)
    print(f"  {who}: {'OK' if not bad else f'{bad}/{N // 4096} wrong'} (want {b}, first {host[0]})", flush=True)


def b_main():
    ck(cu.cuInit(0), "init")
    mine, theirs = context(0), context(1)          # ordinal 0 = B's GPU, ordinal 1 = A's GPU
    for a, b in ((mine, theirs), (theirs, mine)):
        ck(cu.cuCtxSetCurrent(a), "set"); rc = cu.cuCtxEnablePeerAccess(b, 0)
        if rc not in (0, 704):  # 704: already enabled
            ck(rc, "enable peer")
    y = alloc(mine); x = alloc(theirs)
    for label, fn in (("cuMemcpyDtoD (UVA)", lambda d, s: cu.cuMemcpyDtoD_v2(d, s, ctypes.c_size_t(N))),
                      ("cuMemcpyPeer", None)):
        fill(mine, y, 0xB1); fill(theirs, x, 0)
        ck(cu.cuCtxSetCurrent(mine), "set")
        t = time.perf_counter()
        ck(fn(x, y) if fn else cu.cuMemcpyPeer(x, theirs, y, mine, ctypes.c_size_t(N)), "copy")
        ck(cu.cuCtxSynchronize(), "sync"); dt = time.perf_counter() - t
        print(f"B {label}: its GPU -> A's GPU {N / dt / 1e9:.1f} GB/s", flush=True)
        check(theirs, x, 0xB1, f"{label}, X on A's GPU")
    fill(theirs, x, 0x3C)
    ck(cu.cuCtxSetCurrent(theirs), "set")
    h = Handle(); ck(cu.cuIpcGetMemHandle(ctypes.byref(h), x), "ipcget")
    open(SHM, "wb").write(ctypes.string_at(ctypes.addressof(h), 64))
    open(SHM + ".go", "w").write("1")
    while os.path.exists(SHM + ".go"):
        time.sleep(0.05)                               # A reads X and writes 0x7E into it
    fill(mine, y, 0)
    ck(cu.cuCtxSetCurrent(mine), "set")
    t = time.perf_counter(); ck(cu.cuMemcpyDtoD_v2(y, x, ctypes.c_size_t(N)), "pull"); ck(cu.cuCtxSynchronize(), "sync")
    print(f"B pulls what A wrote, A's GPU -> its GPU: {N / (time.perf_counter() - t) / 1e9:.1f} GB/s", flush=True)
    check(mine, y, 0x7E, "Y on B's GPU after the pull")


def a_main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1,0"}
    b = subprocess.Popen([sys.executable, __file__, "B"], env=env)
    while not os.path.exists(SHM + ".go"):
        time.sleep(0.05)
        if b.poll() is not None:
            raise SystemExit("B exited")
    ck(cu.cuInit(0), "init")
    c = context(0)                                     # A's GPU (GPU0)
    ck(cu.cuCtxSetCurrent(c), "set")
    raw = open(SHM, "rb").read(); h = Handle(); ctypes.memmove(ctypes.addressof(h), raw, 64)
    cu.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), Handle, ctypes.c_uint]
    x = ctypes.c_uint64(); ck(cu.cuIpcOpenMemHandle_v2(ctypes.byref(x), h, 1), "ipcopen (same GPU)")
    check(c, x, 0x3C, "A reads X (B's allocation on A's GPU)")
    fill(c, x, 0x7E)
    ck(cu.cuIpcCloseMemHandle(x), "close")
    os.unlink(SHM + ".go")
    b.wait(); os.unlink(SHM)


if __name__ == "__main__":
    b_main() if sys.argv[1:] == ["B"] else a_main()
