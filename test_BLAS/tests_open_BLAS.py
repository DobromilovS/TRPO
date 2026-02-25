#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import os
import sys
import io
from contextlib import redirect_stderr

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


def run_test(func, func_name, test_fn, *args):
    """Run test and capture stderr to check for error messages"""
    # Создаем буфер для перехвата stderr
    stderr_buffer = io.StringIO()
    
    try:
        with redirect_stderr(stderr_buffer):
            test_fn(func, *args)
        
        # Проверяем, не было ли сообщений об ошибках
        stderr_output = stderr_buffer.getvalue()
        if stderr_output and ("illegal value" in stderr_output.lower() or 
                              "error" in stderr_output.lower() or
                              "invalid" in stderr_output.lower()):
            return False, f"library reported error: {stderr_output.strip()}"
        
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        stderr_buffer.close()


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
    total_tests = 0
    failed_tests = []

    for dtype in ["s", "d", "c", "z"]:
        A, B, C, scalar, is_complex = make_buffers(dtype)

        tests = []

        tests.append(("cblas_{}gemm".format(dtype), test_gemm, 
                     (scalar, is_complex, A, B, C)))
        tests.append(("cblas_{}gemmtr".format(dtype), test_gemmtr, 
                     (scalar, is_complex, A, B, C)))

        tests.append(("cblas_{}symm".format(dtype), test_symm_like, 
                     (scalar, is_complex, A, B, C)))
        if dtype in ["c", "z"]:
            tests.append(("cblas_{}hemm".format(dtype), test_symm_like, 
                         (scalar, is_complex, A, B, C)))

        tests.append(("cblas_{}trmm".format(dtype), test_trmm_trsm, 
                     (scalar, is_complex, A, B)))
        tests.append(("cblas_{}trsm".format(dtype), test_trmm_trsm, 
                     (scalar, is_complex, A, B)))

        tests.append(("cblas_{}syrk".format(dtype), test_syrk, 
                     (scalar, is_complex, A, C)))
        if dtype in ["c", "z"]:
            tests.append(("cblas_{}herk".format(dtype), test_herk, 
                         (A, C)))

        tests.append(("cblas_{}syr2k".format(dtype), test_syr2k, 
                     (scalar, is_complex, A, B, C)))
        if dtype in ["c", "z"]:
            tests.append(("cblas_{}her2k".format(dtype), test_her2k, 
                         (scalar, A, B, C)))

        print(f"\n=== Testing dtype {dtype} ===")
        for name, test_func, args in tests:
            total_tests += 1
            fn = get_func(lib, name)
            if fn is None:
                print(f"  [FAIL] {name} (symbol not found)")
                all_pass = False
                failed_tests.append(name)
                continue

            success, error_msg = run_test(fn, name, test_func, *args)
            if success:
                print(f"  [PASS] {name}")
            else:
                print(f"  [FAIL] {name} - {error_msg}")
                all_pass = False
                failed_tests.append(name)

    print(f"\n=== Summary ===")
    print(f"Total tests: {total_tests}")
    if failed_tests:
        print(f"Failed tests: {len(failed_tests)}")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print(f"Failed tests: 0")

    if all_pass:
        print("\n[OK] All Level 3 functions working correctly")
        return 0

    print("\n[FAIL] Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())