import numpy as np
from ctypes import cdll, c_void_p, c_ubyte, c_int, c_float, POINTER, byref

lib = cdll.LoadLibrary('src/liblsdslam.so')

def _bp(arr):
    return arr.ctypes.data_as(POINTER(c_ubyte))

def _fp(arr):
    return arr.ctypes.data_as(POINTER(c_float))

size = lib.get_imagewidth(), lib.get_imageheight()

lib.allocate_tracker.restype = c_void_p

class Tracker(object):
    def __init__(self):
        self.obj = lib.allocate_tracker()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        lib.release_tracker(c_void_p(self.obj))

    def init(self,
            initial_D=1.0,
            initial_V=1e5,
            mask_thresh=0.1,
            huber_delta=3,
            K=np.eye(3, dtype=np.float32),
            eps=0.001,
            max_iter=100
            ):
        lib.tracker_init(
                c_void_p(self.obj),
                c_float(initial_D),
                c_float(initial_V),
                c_float(mask_thresh),
                c_float(huber_delta),
                _fp(K),
                c_float(eps),
                c_int(max_iter)
                )

    def estimate(self, image):
        n = np.zeros(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        lib.tracker_estimate(
                c_void_p(self.obj),
                _bp(image),
                _fp(n), _fp(t))
        return n, t

