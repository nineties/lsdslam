import numpy as np
from util import assert_allclose, repeat
import lsdslam.lib as L

@repeat(100)
def test_mul3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    B = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        A.dot(B),
        L.mul3x3(A, B)
        )

@repeat(100)
def test_det3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        np.linalg.det(A),
        L.det3x3(A)
        )

@repeat(100)
def test_inv3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        np.linalg.inv(A),
        L.inv3x3(A)
        )
