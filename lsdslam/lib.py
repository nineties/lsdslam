import numpy as np
from ctypes import cdll, c_float, POINTER

lib = cdll.LoadLibrary('src/liblsdslam.so')

def _float_ptr(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def mul3x3(A, B):
    C = np.zeros_like(A)
    lib.mul3x3(_float_ptr(C), _float_ptr(A), _float_ptr(B))
    return C

lib.det3x3.restype = c_float
def det3x3(A):
    return lib.det3x3(_float_ptr(A))

def inv3x3(A):
    B = np.zeros_like(A)
    lib.inv3x3(_float_ptr(B), _float_ptr(A))
    return B

def affine3d(A, b, x):
    y = np.zeros_like(x)
    lib.affine3d(_float_ptr(y), _float_ptr(A), _float_ptr(b), _float_ptr(x))
    return y

def R(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.compute_R(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def R_theta(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.compute_R_theta(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def R_n(n, theta):
    y = np.zeros((3, 3, 3), dtype=np.float32)
    lib.compute_R_n(_float_ptr(y), _float_ptr(n), c_float(theta))
    return y

def T(t, n, theta, rho, x):
    y = np.zeros(3, dtype=np.float32)
    lib.T(_float_ptr(y), _float_ptr(t), _float_ptr(n), c_float(theta), c_float(rho), _float_ptr(x))
    return y

def T_t(t, n, theta, rho, x):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.T_t(_float_ptr(y), _float_ptr(t), _float_ptr(n), c_float(theta), c_float(rho), _float_ptr(x))
    return y
