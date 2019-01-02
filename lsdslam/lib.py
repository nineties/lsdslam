import numpy as np
from ctypes import cdll, c_float, POINTER

lib = cdll.LoadLibrary('src/liblsdslam.so')

def _float_ptr(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def R(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.R(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def R_theta(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.R_theta(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def R_n(n, theta):
    y = np.zeros((3, 3, 3), dtype=np.float32)
    lib.R_n(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def T(t, n, theta, rho, x):
    y = np.zeros(3, dtype=np.float32)
    lib.T(_float_ptr(y), _float_ptr(t), _float_ptr(n), c_float(theta), c_float(rho), _float_ptr(x))
    return y

def T_t(t, n, theta, rho, x):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.T_t(_float_ptr(y), _float_ptr(t), _float_ptr(n), c_float(theta), c_float(rho), _float_ptr(x))
    return y
