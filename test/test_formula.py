import time
from nose.tools import make_decorator
import numpy as np
from sympy import Matrix, symbols, diff, lambdify, cos, sin

import lsdslam.formula as F

# Utilities

def assert_allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-3)

def repeat(n):
    def decorate(f):
        def newfunc(*args, **kw):
            for i in range(n):
                f(*args, **kw)
        return make_decorator(f)(newfunc)
    return decorate

# Formulas
x = Matrix(symbols('x1 x2 x3')) # 3d point
p = Matrix(symbols('u v'))      # 2d point
n = Matrix(symbols('n1 n2 n3')) # axis of rotation
theta = symbols('theta')        # angle of rotation
t = Matrix(symbols('t1 t2 t3')) # translation vector
rho = symbols('rho')            # scale factor = exp(rho)

# Rodrigues's rotation formula
def rotate_formula(n, theta, x):
    return cos(theta)*x + sin(theta)*(n.cross(x)) + (1-cos(theta))*(n.dot(x))*n

rotate = lambdify((n, theta, x), rotate_formula(n, theta, x))

# d(rotate)/d(theta)
rotate_theta = lambdify((n, theta, x), diff(rotate_formula(n, theta, x), theta))

# d(rotate)/d(n)
rotate_n = lambdify((n, theta, x), diff(rotate_formula(n, theta, x), n))

@repeat(100)
def test_rotate():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
            rotate(n, theta, x).flatten(),
            F.rotate(n, theta, x)
            )

@repeat(100)
def test_rotate_theta():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
        rotate_theta(n, theta, x).flatten(),
        F.rotate_theta(n, theta, x)
        )

@repeat(100)
def test_rotate_n():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
        np.array(rotate_n(n, theta, x)).reshape(3, 3),
        F.rotate_n(n, theta, x)
        )

