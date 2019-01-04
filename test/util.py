from nose.tools import make_decorator
import numpy as np
from PIL import Image
import lib as L

# Utilities

def assert_allclose(x, y, rtol=1e-3, atol=0):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

def repeat(n):
    def decorate(f):
        def newfunc(*args, **kw):
            for i in range(n):
                f(*args, **kw)
        return make_decorator(f)(newfunc)
    return decorate

def random_vec(n):
    return np.random.randn(n).astype(np.float32)

def random_norm(n):
    v = random_vec(n)
    return v/np.sqrt(v.dot(v))

def read_image(path, dtype=np.float32):
    return np.asarray(Image.open(path).convert('P').resize(L.SIZE), dtype=dtype)

