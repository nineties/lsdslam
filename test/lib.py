import numpy as np
from ctypes import cdll, c_bool, c_int, c_float, POINTER, Structure, byref, memmove

lib = cdll.LoadLibrary('src/liblsdslam_test.so')

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

def sR(s, n):
    y = np.zeros((3, 3), dtype=np.float32)
    lib.compute_sR(_fp(y), c_float(s), _fp(n))
    return y

def sR_n(s, n):
    y = np.zeros((3, 3, 3), dtype=np.float32)
    lib.compute_sR_n(_fp(y), c_float(s), _fp(n))
    return y

def identity():
    rho = c_float()
    n = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    lib.compute_identity(byref(rho), _fp(n), _fp(t))
    return rho.value, n, t

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

def gradu(I):
    y = np.zeros_like(I)
    lib.gradu(_fp(y), _fp(I))
    return y

def gradv(I):
    y = np.zeros_like(I)
    lib.gradv(_fp(y), _fp(I))
    return y

lib.variance.restype = c_float
def variance(I):
    return lib.variance(_fp(I))

def solve(dof, A, b):
    x = np.zeros_like(b)
    A = A.copy()
    b = b.copy()
    lib.solve(_fp(x), c_int(dof), _fp(A), _fp(b))
    return x

class Param(Structure):
    _fields_ = [
            ('initial_D', c_float),
            ('initial_V', c_float),
            ('mask_thresh', c_float),
            ('huber_delta', c_float),
            ('K', c_float * 3 * 3)
            ]
    def __init__(self, initial_D, initial_V, mask_thresh, huber_delta, K):
        self.initial_D = initial_D
        self.initial_V = initial_V
        self.mask_thresh = mask_thresh
        self.huber_delta = huber_delta

        memmove(self.K, _fp(K), K.nbytes)

def BFGS_update(dof, H, y, s):
    lib.BFGS_update(dof, _fp(H), _fp(y), _fp(s))
