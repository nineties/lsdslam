import numpy as np
from ctypes import cdll, c_float, POINTER

lib = cdll.LoadLibrary('src/liblsdslam.so')

def _float_ptr(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def R(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros_like(x)
    lib.R(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y

def R_theta(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros_like(x)
    lib.R_theta(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y

def R_n(n, theta, x):
    assert(x.dtype == np.float32)
    assert(n.dtype == np.float32)
    y = np.zeros((3, 3), dtype=np.float32)
    lib.R_n(_float_ptr(y), _float_ptr(n), c_float(theta), _float_ptr(x))
    return y
