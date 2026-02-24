#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import os
import sys

CblasRowMajor = 101

CblasNoTrans = 111
CblasUpper = 121

CblasNonUnit = 131
CblasLeft = 141

class ComplexFloat(ctypes.Structure):
    _fields_ = [("re", ctypes.c_float), ("im", ctypes.c_float)]


class ComplexDouble(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


def load_openblas(path_from_user):
    if path_from_user:
        return ctypes.CDLL(path_from_user)

    name = ctypes.util.find_library("openblas")
    if name:
        return ctypes.CDLL(name)

    for name2 in ["libopenblas.so", "libopenblas.dylib", "openblas.dll"]:
        try:
            return ctypes.CDLL(name2)
        except OSError:
            pass

    raise RuntimeError("Could not find OpenBLAS. Specify path with --lib")


def get_func(lib, func_name):
    try:
        return getattr(lib, func_name)
    except AttributeError:
        return None


def ptr(x):
    return ctypes.cast(x, ctypes.c_void_p)


def make_buffers(dtype):
    if dtype == "s":
        A = (ctypes.c_float * 1)(0.0)
        B = (ctypes.c_float * 1)(0.0)
        C = (ctypes.c_float * 1)(0.0)
        scalar = ctypes.c_float
        is_complex = False
        return A, B, C, scalar, is_complex

    if dtype == "d":
        A = (ctypes.c_double * 1)(0.0)
        B = (ctypes.c_double * 1)(0.0)
        C = (ctypes.c_double * 1)(0.0)
        scalar = ctypes.c_double
        is_complex = False
        return A, B, C, scalar, is_complex

    if dtype == "c":
        A = (ComplexFloat * 1)(ComplexFloat(0.0, 0.0))
        B = (ComplexFloat * 1)(ComplexFloat(0.0, 0.0))
        C = (ComplexFloat * 1)(ComplexFloat(0.0, 0.0))
        scalar = ComplexFloat
        is_complex = True
        return A, B, C, scalar, is_complex

    if dtype == "z":
        A = (ComplexDouble * 1)(ComplexDouble(0.0, 0.0))
        B = (ComplexDouble * 1)(ComplexDouble(0.0, 0.0))
        C = (ComplexDouble * 1)(ComplexDouble(0.0, 0.0))
        scalar = ComplexDouble
        is_complex = True
        return A, B, C, scalar, is_complex

    raise ValueError("dtype must be s/d/c/z")


def zero(scalar_type, is_complex):
    if is_complex:
        return scalar_type(0.0, 0.0)
    return scalar_type(0)

def test_gemm(fn, scalar, is_complex, A, B, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       0, 0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       ptr(B), 1,
       zero(scalar, is_complex),
       ptr(C), 1)


def test_gemmtr(fn, scalar, is_complex, A, B, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
       0, 0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       ptr(B), 1,
       zero(scalar, is_complex),
       ptr(C), 1)


def test_symm_like(fn, scalar, is_complex, A, B, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasLeft, CblasUpper,
       0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       ptr(B), 1,
       zero(scalar, is_complex),
       ptr(C), 1)


def test_trmm_trsm(fn, scalar, is_complex, A, B):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
       0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       ptr(B), 1)


def test_syrk(fn, scalar, is_complex, A, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasUpper, CblasNoTrans,
       0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       zero(scalar, is_complex),
       ptr(C), 1)


def test_herk(fn, A, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasUpper, CblasNoTrans,
       0, 0,
       0.0,
       ptr(A), 1,
       0.0,
       ptr(C), 1)


def test_syr2k(fn, scalar, is_complex, A, B, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        scalar,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasUpper, CblasNoTrans,
       0, 0,
       zero(scalar, is_complex),
       ptr(A), 1,
       ptr(B), 1,
       zero(scalar, is_complex),
       ptr(C), 1)


def test_her2k(fn, scalar_complex, A, B, C):
    fn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar_complex,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_int
    ]
    fn.restype = None

    fn(CblasRowMajor, CblasUpper, CblasNoTrans,
       0, 0,
       scalar_complex(0.0, 0.0),
       ptr(A), 1,
       ptr(B), 1,
       0.0,
       ptr(C), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default=None, help="Path to libopenblas (e.g. /usr/lib/libopenblas.so)")
    parser.add_argument("--threads", type=int, default=None, help="OPENBLAS_NUM_THREADS")
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    try:
        lib = load_openblas(args.lib)
    except Exception as e:
        print("[FAIL] Could not load OpenBLAS:", e)
        return 2

    all_pass = True

    for dtype in ["s", "d", "c", "z"]:
        A, B, C, scalar, is_complex = make_buffers(dtype)

        tests = []

        tests.append((f"cblas_{dtype}gemm",   lambda fn: test_gemm(fn, scalar, is_complex, A, B, C)))
        tests.append((f"cblas_{dtype}gemmtr", lambda fn: test_gemmtr(fn, scalar, is_complex, A, B, C)))

        tests.append((f"cblas_{dtype}symm", lambda fn: test_symm_like(fn, scalar, is_complex, A, B, C)))
        if dtype in ["c", "z"]:
            tests.append((f"cblas_{dtype}hemm", lambda fn: test_symm_like(fn, scalar, is_complex, A, B, C)))

        tests.append((f"cblas_{dtype}trmm", lambda fn: test_trmm_trsm(fn, scalar, is_complex, A, B)))
        tests.append((f"cblas_{dtype}trsm", lambda fn: test_trmm_trsm(fn, scalar, is_complex, A, B)))

        tests.append((f"cblas_{dtype}syrk", lambda fn: test_syrk(fn, scalar, is_complex, A, C)))
        if dtype in ["c", "z"]:
            tests.append((f"cblas_{dtype}herk", lambda fn: test_herk(fn, A, C)))

        tests.append((f"cblas_{dtype}syr2k", lambda fn: test_syr2k(fn, scalar, is_complex, A, B, C)))
        if dtype in ["c", "z"]:
            tests.append((f"cblas_{dtype}her2k", lambda fn: test_her2k(fn, scalar, A, B, C)))

        print(f"\ndtype {dtype}")
        for name, caller in tests:
            fn = get_func(lib, name)
            if fn is None:
                print("[FAIL]", name, "(symbol not found)")
                all_pass = False
                continue

            try:
                caller(fn)
                print("[PASS]", name)
            except Exception as e:
                print("[FAIL]", name, "-", type(e).__name__, e)
                all_pass = False

    if all_pass:
        print("\n[OK] All Level 3 functions found and called without errors")
        return 0

    print("\n[FAIL] Errors encountered (symbol not found or exception during call)")
    return 1


if __name__ == "__main__":
    sys.exit(main())