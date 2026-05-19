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

    for candidate in ("libopenblas.so", "libopenblas.dylib", "openblas.dll"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            pass

    raise RuntimeError("Could not find OpenBLAS. Specify path with --lib")


def symbol(lib, name):
    try:
        return getattr(lib, name)
    except AttributeError as exc:
        raise RuntimeError(f"symbol {name} not found") from exc


def as_void_p(value):
    return ctypes.cast(value, ctypes.c_void_p)


def close_enough(actual, expected, eps=1e-5):
    return abs(float(actual) - float(expected)) <= eps


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_real_tests(lib, prefix, scalar_type, eps):
    scalar_ptr = ctypes.POINTER(scalar_type)
    cases = []

    gemm = symbol(lib, f"cblas_{prefix}gemm")
    gemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
        scalar_ptr, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
    ]
    a = (scalar_type * 1)(2)
    b = (scalar_type * 1)(3)
    c = (scalar_type * 1)(0)
    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1,
         scalar_type(1), a, 1, b, 1, scalar_type(0), c, 1)
    require(close_enough(c[0], 6, eps), f"{prefix}gemm expected 6, got {c[0]}")
    cases.append(f"cblas_{prefix}gemm")

    symm = symbol(lib, f"cblas_{prefix}symm")
    symm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
        scalar_ptr, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
    ]
    c = (scalar_type * 1)(0)
    symm(CblasRowMajor, CblasLeft, CblasUpper, 1, 1,
         scalar_type(1), a, 1, b, 1, scalar_type(0), c, 1)
    require(close_enough(c[0], 6, eps), f"{prefix}symm expected 6, got {c[0]}")
    cases.append(f"cblas_{prefix}symm")

    trmm = symbol(lib, f"cblas_{prefix}trmm")
    trmm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
        scalar_ptr, ctypes.c_int,
    ]
    b_tri = (scalar_type * 1)(3)
    trmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
         1, 1, scalar_type(1), a, 1, b_tri, 1)
    require(close_enough(b_tri[0], 6, eps), f"{prefix}trmm expected 6, got {b_tri[0]}")
    cases.append(f"cblas_{prefix}trmm")

    trsm = symbol(lib, f"cblas_{prefix}trsm")
    trsm.argtypes = trmm.argtypes
    b_solve = (scalar_type * 1)(6)
    trsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
         1, 1, scalar_type(1), a, 1, b_solve, 1)
    require(close_enough(b_solve[0], 3, eps), f"{prefix}trsm expected 3, got {b_solve[0]}")
    cases.append(f"cblas_{prefix}trsm")

    syrk = symbol(lib, f"cblas_{prefix}syrk")
    syrk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
    ]
    c = (scalar_type * 1)(0)
    syrk(CblasRowMajor, CblasUpper, CblasNoTrans, 1, 1,
         scalar_type(1), b, 1, scalar_type(0), c, 1)
    require(close_enough(c[0], 9, eps), f"{prefix}syrk expected 9, got {c[0]}")
    cases.append(f"cblas_{prefix}syrk")

    syr2k = symbol(lib, f"cblas_{prefix}syr2k")
    syr2k.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
        scalar_ptr, ctypes.c_int,
        scalar_type,
        scalar_ptr, ctypes.c_int,
    ]
    c = (scalar_type * 1)(0)
    syr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 1, 1,
          scalar_type(1), a, 1, b, 1, scalar_type(0), c, 1)
    require(close_enough(c[0], 12, eps), f"{prefix}syr2k expected 12, got {c[0]}")
    cases.append(f"cblas_{prefix}syr2k")

    return cases


def run_complex_gemm_test(lib, prefix, complex_type, eps):
    gemm = symbol(lib, f"cblas_{prefix}gemm")
    gemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
    ]

    alpha = complex_type(1, 0)
    beta = complex_type(0, 0)
    a = (complex_type * 1)(complex_type(2, 1))
    b = (complex_type * 1)(complex_type(3, -1))
    c = (complex_type * 1)(complex_type(0, 0))

    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1,
         as_void_p(ctypes.byref(alpha)),
         as_void_p(a), 1,
         as_void_p(b), 1,
         as_void_p(ctypes.byref(beta)),
         as_void_p(c), 1)

    require(close_enough(c[0].re, 7, eps), f"{prefix}gemm real expected 7, got {c[0].re}")
    require(close_enough(c[0].im, 1, eps), f"{prefix}gemm imag expected 1, got {c[0].im}")
    return [f"cblas_{prefix}gemm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default=None, help="Path to libopenblas")
    parser.add_argument("--threads", type=int, default=1, help="OPENBLAS_NUM_THREADS")
    args = parser.parse_args()

    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    try:
        lib = load_openblas(args.lib)
        passed = []
        passed.extend(run_real_tests(lib, "s", ctypes.c_float, 1e-4))
        passed.extend(run_real_tests(lib, "d", ctypes.c_double, 1e-9))
        passed.extend(run_complex_gemm_test(lib, "c", ComplexFloat, 1e-4))
        passed.extend(run_complex_gemm_test(lib, "z", ComplexDouble, 1e-9))
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("=== Interface tests ===")
    for name in passed:
        print(f"  [PASS] {name}")
    print(f"\n[OK] {len(passed)} OpenBLAS interface checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
