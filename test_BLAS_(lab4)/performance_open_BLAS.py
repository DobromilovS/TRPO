#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import os
import sys
import time

CblasRowMajor = 101
CblasNoTrans = 111


def load_openblas(path_from_user):
    if path_from_user:
        return ctypes.CDLL(path_from_user)

    name = ctypes.util.find_library("openblas")
    if name:
        return ctypes.CDLL(name)

    for candidate in ("libopenblas.so", "libopenblas.dylib", "openblas.dll"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            pass

    raise RuntimeError("Could not find OpenBLAS. Specify path with --lib")


def make_matrix(size, seed):
    count = size * size
    return (ctypes.c_double * count)(*[
        ((i * 17 + seed) % 101) / 101.0
        for i in range(count)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default=None, help="Path to libopenblas")
    parser.add_argument("--size", type=int, default=128, help="Square matrix size")
    parser.add_argument("--repeat", type=int, default=5, help="Number of benchmark repetitions")
    parser.add_argument("--threads", type=int, default=1, help="OPENBLAS_NUM_THREADS")
    parser.add_argument("--min-gflops", type=float, default=0.01, help="Minimal acceptable GFLOPS")
    args = parser.parse_args()

    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    try:
        lib = load_openblas(args.lib)
        dgemm = lib.cblas_dgemm
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    dgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    dgemm.restype = None

    n = args.size
    a = make_matrix(n, 1)
    b = make_matrix(n, 7)
    c = (ctypes.c_double * (n * n))()

    # Warm-up run.
    dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          n, n, n, 1.0, a, n, b, n, 0.0, c, n)

    start = time.perf_counter()
    for _ in range(args.repeat):
        dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, n, n, 1.0, a, n, b, n, 0.0, c, n)
    elapsed = time.perf_counter() - start

    operations = 2.0 * n * n * n * args.repeat
    gflops = operations / elapsed / 1_000_000_000

    print("=== Performance test ===")
    print(f"matrix size: {n}x{n}")
    print(f"repetitions: {args.repeat}")
    print(f"elapsed: {elapsed:.6f} s")
    print(f"performance: {gflops:.4f} GFLOPS")

    if c[0] == 0:
        print("[FAIL] result matrix was not updated")
        return 1

    if gflops < args.min_gflops:
        print(f"[FAIL] performance below threshold: {gflops:.4f} < {args.min_gflops}")
        return 1

    print("[OK] performance threshold passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
