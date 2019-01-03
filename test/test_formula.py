import numpy as np
from sympy import Matrix, symbols, diff, simplify, lambdify, transpose, cos, sin, exp, eye

from util import assert_allclose, repeat, random_vec, random_norm
import lsdslam.lib as L

# Formulas
x1, x2, x3 = symbols('x1 x2 x3')
x = Matrix([x1, x2, x3])        # 3d point
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

def T_formula(rho, n, theta, t, x):
    return exp(rho)*R_formula(n, theta)*x + t

T = lambdify((rho, n, theta, t, x), T_formula(rho, n, theta, t, x))

def pi_formula(x1, x2, x3):
    return Matrix([x1/x3, x2/x3])

pi = lambdify((x,), pi_formula(x1, x2, x3))

def pi_x_formula(x1, x2, x3):
    f = pi_formula(x1, x2, x3)
    return diff(f, x1).row_join(diff(f, x2)).row_join(diff(f, x3))

pi_x = lambdify((x,), pi_x_formula(x1, x2, x3))

@repeat(100)
def test_R():
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
            R(n, theta),
            L.R(n, theta)
            )

@repeat(100)
def test_R_theta():
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
            R_theta(n, theta),
            L.R_theta(n, theta)
            )

@repeat(100)
def test_R_n():
    n = random_norm(3)
    theta = np.random.randn()

    A = L.R_n(n, theta)

    for i in range(3):
        assert_allclose(R_n[i](n, theta), A[i])

@repeat(100)
def test_rotate_n():
    n = random_norm(3)
    x = random_vec(3)
    theta = np.random.randn()

    A = L.R_n(n, theta)
    for i in range(3):
        assert_allclose(
            rotate_n[i](n, theta, x).flatten(),
            A[i].dot(x)
            )

@repeat(100)
def test_T():
    x = random_vec(3)
    n = random_norm(3)
    t = random_vec(3)
    theta = np.random.randn()
    rho = np.random.randn()

    A, b = L.precompute_T(rho, n, theta, t)
    assert_allclose(
            T(rho, n, theta, t, x).flatten(),
            A.dot(x) + b
            )

@repeat(100)
def test_KT():
    x = random_vec(3)
    n = random_norm(3)
    t = random_vec(3)
    theta = np.random.randn()
    rho = np.random.randn()
    K = np.random.randn(3, 3).astype(np.float32)

    A, b = L.precompute_KT(K, rho, n, theta, t)
    assert_allclose(
            K.dot(T(rho, n, theta, t, x).flatten()),
            A.dot(x) + b
            )

@repeat(100)
def test_pi():
    x = random_vec(3)

    assert_allclose(
            pi(x).flatten(),
            L.pi(x)
            )

@repeat(100)
def test_pi_x():
    x = random_vec(3)

    assert_allclose(
            pi_x(x),
            L.pi_x(x)
            )
