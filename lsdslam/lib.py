import numpy as np
from ctypes import cdll, c_bool, c_int, c_float, POINTER, Structure, byref

lib = cdll.LoadLibrary('src/liblsdslam.so')

lib.get_imagewidth.restype = c_int
lib.get_imageheight.restype = c_int

# Image size
WIDTH = lib.get_imagewidth()
HEIGHT = lib.get_imageheight()
SIZE = (WIDTH, HEIGHT)

def _fp(arr):
    return arr.ctypes.data_as(POINTER(c_float))

def mul3x3(A, B):
    C = np.zeros_like(A)
    lib.mul3x3(_fp(C), _fp(A), _fp(B))
    return C

lib.det3x3.restype = c_float
def det3x3(A):
    return lib.det3x3(_fp(A))

def inv3x3(A):
    B = np.zeros_like(A)
    lib.inv3x3(_fp(B), _fp(A))
    return B

def affine3d(A, b, x):
    y = np.zeros_like(x)
    lib.affine3d(_fp(y), _fp(A), _fp(b), _fp(x))
    return y

def R(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.compute_R(_fp(y), _fp(n), c_float(theta))
    return y

def R_theta(n, theta):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.compute_R_theta(_fp(y), _fp(n), c_float(theta))
    return y

def R_n(n, theta):
    y = np.zeros((3, 3, 3), dtype=np.float32)
    lib.compute_R_n(_fp(y), _fp(n), c_float(theta))
    return y

def precompute_T(rho, n, theta, t):
    A = np.zeros((3, 3), dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    lib.precompute_T(_fp(A), _fp(b), c_float(rho), _fp(n), c_float(theta), _fp(t))
    return A, b

def precompute_KT(K, rho, n, theta, t):
    A = np.zeros((3, 3), dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    lib.precompute_KT(_fp(A), _fp(b), _fp(K), c_float(rho), _fp(n), c_float(theta), _fp(t))
    return A, b

def pi(x):
    y = np.zeros(3, dtype=np.float32)
    lib.pi(_fp(y), _fp(x))
    return y

def pip_x(x):
    y = np.zeros((2, 3), dtype=np.float32)
    lib.pip_x(_fp(y), _fp(x))
    return y

def piinv(p, d):
    y = np.zeros(3, dtype=np.float32)
    lib.piinv(_fp(y), _fp(p), c_float(d))
    return y

def piinv_d(p, d):
    y = np.zeros(3, dtype=np.float32)
    lib.piinv_d(_fp(y), _fp(p), c_float(d))
    return y

def filter3x3(K, I):
    y = np.zeros_like(I)
    lib.filter3x3(_fp(y), _fp(K), _fp(I))
    return y

def gaussian_filter3x3(I):
    y = np.zeros_like(I)
    lib.gaussian_filter3x3(_fp(y), _fp(I))
    return y

def sobelx(I):
    y = np.zeros_like(I)
    lib.sobelx(_fp(y), _fp(I))
    return y

def sobely(I):
    y = np.zeros_like(I)
    lib.sobely(_fp(y), _fp(I))
    return y

class Param(Structure):
    _fields_ = [
            ('mask_thresh', c_float)
            ]

class ComputeCache(Structure):
    _fields_ = [
            ('mask', c_bool * WIDTH * HEIGHT),
            ('Iref', c_float * WIDTH * HEIGHT),
            ('Dref', c_float * WIDTH * HEIGHT),
            ('Vref', c_float * WIDTH * HEIGHT),
            ('I',   c_float * WIDTH * HEIGHT),
            ('I_u', c_float * WIDTH * HEIGHT),
            ('I_v', c_float * WIDTH * HEIGHT),
            ]

class LSDSLAMStruct(Structure):
    _fields_ = [
            ('param', Param),
            ('cache', ComputeCache)
            ]

def precompute_cache(
        slam,
        Iref, Dref, Vref,
        I
        ):
    lib.precompute_cache(
            byref(slam),
            _fp(Iref), _fp(Dref), _fp(Vref), _fp(I)
            )

# Photometric Residual
def rp(slam, p):
    u, v = p
    pass