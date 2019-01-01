import numpy as np
from ctypes import cdll, c_float, POINTER

lib = cdll.LoadLibrary('src/liblsdslam.so')

def _float_ptr(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def rotate(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros_like(x)
    lib.rotate(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y

def rotate_theta(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros_like(x)
    lib.rotate_theta(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y

def rotate_n(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros((3, 3), dtype=np.float32)
    lib.rotate_n(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y
