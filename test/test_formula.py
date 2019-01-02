import time
from nose.tools import make_decorator
import numpy as np
from sympy import Matrix, symbols, diff, simplify, lambdify, transpose, cos, sin, exp, eye

import lsdslam.formula as F

# Utilities

def assert_allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)

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

# Formulas
x = Matrix(symbols('x1 x2 x3')) # 3d point
p = Matrix(symbols('u v'))      # 2d point
n1, n2, n3 = symbols('n1 n2 n3')
n = Matrix([n1, n2, n3])        # axis of rotation
theta = symbols('theta')        # angle of rotation
t = Matrix(symbols('t1 t2 t3')) # translation vector
rho = symbols('rho')            # scale factor = exp(rho)

# Rodrigues's rotation formula
def R_formula(n, theta):
    N = Matrix([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    return eye(3) + sin(theta)*N + (1-cos(theta))*N**2

R = lambdify((n, theta), R_formula(n, theta))

# dR/d(theta)
R_theta = lambdify((n, theta), diff(R_formula(n, theta), theta))

# dR/dn
R_n = [
    lambdify((n, theta), diff(R_formula(n, theta), n1)),
    lambdify((n, theta), diff(R_formula(n, theta), n2)),
    lambdify((n, theta), diff(R_formula(n, theta), n3))
    ]

def rotate_formula(n, theta, x):
    return R_formula(n, theta) * x

def rotate_n_formula(n, theta, x):
    y = rotate_formula(n, theta, x)
    return diff(y, n_1).row_join(diff(y, n_2)).row_join(diff(y, n_3))

rotate_n = [
    lambdify((n, theta, x), diff(rotate_formula(n, theta, x), n1)),
    lambdify((n, theta, x), diff(rotate_formula(n, theta, x), n2)),
    lambdify((n, theta, x), diff(rotate_formula(n, theta, x), n3))
    ]

@repeat(100)
def test_R():
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
            R(n, theta),
            F.R(n, theta)
            )

@repeat(100)
def test_R_theta():
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
        R_theta(n, theta),
        F.R_theta(n, theta)
        )

@repeat(100)
def test_R_n():
    n = random_norm(3)
    theta = np.random.randn()

    A = F.R_n(n, theta)

    for i in range(3):
        assert_allclose(R_n[i](n, theta), A[i])

@repeat(100)
def test_rotate_n():
    n = random_norm(3)
    x = random_vec(3)
    theta = np.random.randn()

    A = F.R_n(n, theta)
    for i in range(3):
        assert_allclose(
            rotate_n[i](n, theta, x).flatten(),
            A[i].dot(x)
            )
