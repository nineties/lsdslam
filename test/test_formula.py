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
def R_formula(n, theta, x):
    return cos(theta)*x + sin(theta)*(n.cross(x)) + (1-cos(theta))*(n.dot(x))*n

R = lambdify((n, theta, x), R_formula(n, theta, x))

# d(R)/d(theta)
R_theta = lambdify((n, theta, x), diff(R_formula(n, theta, x), theta))

# d(R)/d(n)
R_n = lambdify((n, theta, x), diff(R_formula(n, theta, x), n))

@repeat(100)
def test_R():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
            R(n, theta, x).flatten(),
            F.R(n, theta, x)
            )

@repeat(100)
def test_R_theta():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
        R_theta(n, theta, x).flatten(),
        F.R_theta(n, theta, x)
        )

@repeat(100)
def test_R_n():
    n = np.random.randn(3).astype(np.float32)
    x = np.random.randn(3).astype(np.float32)
    theta = np.random.randn()

    assert_allclose(
        np.array(R_n(n, theta, x)).reshape(3, 3),
        F.R_n(n, theta, x)
        )

