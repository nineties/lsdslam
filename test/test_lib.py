import time
import numpy as np
from sympy import Matrix, symbols, diff, simplify, lambdify, transpose, cos, sin, exp, sqrt, eye
from scipy.signal import convolve2d
from scipy.ndimage import sobel
from PIL import Image

import lib as L
from util import *

def test_mul3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    B = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        A.dot(B),
        L.mul3x3(A, B)
        )

def test_det3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        np.linalg.det(A),
        L.det3x3(A)
        )

def test_inv3x3():
    A = np.random.randn(3, 3).astype(np.float32)
    assert_allclose(
        np.linalg.inv(A),
        L.inv3x3(A)
        )

def test_affine3d():
    A = np.random.randn(3, 3).astype(np.float32)
    b = random_vec(3)
    x = random_vec(3)

    assert_allclose(
        A.dot(x) + b,
        L.affine3d(A, b, x)
        )

#==== Formulas ====
x1, x2, x3 = symbols('x1 x2 x3')
n1, n2, n3 = symbols('n1 n2 n3')
t1, t2, t3 = symbols('t1 t2 t3')
u, v = symbols('u v')
d = symbols('d')
x = Matrix([x1, x2, x3])   # 3d point
p = Matrix([u, v])         # 2d point
n = Matrix([n1, n2, n3])   # axis of rotation
theta = symbols('theta')   # angle of rotation
t = Matrix([t1, t2, t3])   # translation vector
rho = symbols('rho')       # scale factor = exp(rho)

# Rodrigues's rotation formula
def R_formula(n):
    theta = sqrt(n.dot(n))
    N = Matrix([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    return eye(3) + sin(theta)/theta*N + ((1-cos(theta))/theta**2)*N**2

R = lambdify((n,), R_formula(n))

# dR/d(theta)
R_theta = lambdify((n,), diff(R_formula(n), theta))

# dR/dn
R_n = [
    lambdify((n,), diff(R_formula(n), n1)),
    lambdify((n,), diff(R_formula(n), n2)),
    lambdify((n,), diff(R_formula(n), n3))
    ]

def rotate_formula(n, x):
    return R_formula(n) * x

def rotate_n_formula(n, x):
    y = rotate_formula(n, x)
    return diff(y, n_1).row_join(diff(y, n_2)).row_join(diff(y, n_3))

rotate_n = [
    lambdify((n, x), diff(rotate_formula(n, x), n1)),
    lambdify((n, x), diff(rotate_formula(n, x), n2)),
    lambdify((n, x), diff(rotate_formula(n, x), n3))
    ]

def T_formula(rho, n, t, x):
    return exp(rho)*R_formula(n)*x + t

T = lambdify((rho, n, t, x), T_formula(rho, n, t, x))

def pip_formula(x1, x2, x3):
    return Matrix([x1/x3, x2/x3])

def pid_formula(x3):
    return 1/x3

def pi_formula(x1, x2, x3):
    return pip_formula(x1, x2, x3).col_join(Matrix([pid_formula(x3)]))

pi  = lambdify((x,), pi_formula(x1, x2, x3))
pip = lambdify((x,), pip_formula(x1, x2, x3))

def pip_x_formula(x1, x2, x3):
    f = pip_formula(x1, x2, x3)
    return diff(f, x1).row_join(diff(f, x2)).row_join(diff(f, x3))

pip_x = lambdify((x,), pip_x_formula(x1, x2, x3))

def piinv_formula(u, v, d):
    return Matrix([u/d, v/d, 1/d])

piinv = lambdify((p, d), piinv_formula(u, v, d))
piinv_d = lambdify((p, d), diff(piinv_formula(u, v, d), d))

#==== Tests ====

def test_R():
    n = random_vec(3)
    s = np.random.randn()

    assert_allclose(
            s*R(n),
            L.sR(s, n)
            )

def test_sR_n():
    n = random_norm(3)
    s = np.random.randn()
    A = L.sR_n(s, n)
    for i in range(3):
        assert_allclose(s*R_n[i](n), A[i])

def test_rotate_n():
    n = random_norm(3)
    x = random_vec(3)
    s = np.random.randn()
    A = L.sR_n(s, n)
    for i in range(3):
        assert_allclose(
            s*rotate_n[i](n, x).flatten(),
            A[i].dot(x)
            )

def test_pi():
    x = random_vec(3)

    assert_allclose(
            pi(x).flatten(),
            L.pi(x)
            )

def test_pip_x():
    x = random_vec(3)

    assert_allclose(
            pip_x(x),
            L.pip_x(x)
            )

def test_piinv():
    x = random_vec(2)
    d = np.random.randn()

    assert_allclose(
            piinv(x, d).flatten(),
            L.piinv(x, d)
            )

def test_piinv_d():
    x = random_vec(2)
    d = np.random.randn()

    assert_allclose(
            piinv_d(x, d).flatten(),
            L.piinv_d(x, d)
            )

def test_filter3x3():
    I = read_image('test/I.png')/255.
    K = np.random.randn(3, 3).astype(np.float32)

    assert_allclose(convolve2d(I, K, mode='same'), L.filter3x3(K, I),
            rtol=0, atol=1e-5)

def test_gaussian_filter():
    I = read_image('test/I.png')/255.
    assert_allclose(
            convolve2d(I,
                [[1/9., 1/9., 1/9.],
                 [1/9., 1/9., 1/9.],
                 [1/9., 1/9., 1/9.]], mode='same'),
            L.gaussian_filter3x3(I),
            rtol=0, atol=1e-5
            )

def test_sobel_filter():
    I = read_image('test/I.png')/255.
    assert_allclose(
            sobel(I, axis=0, mode='constant')/4,
            L.gradu(I),
            rtol=0, atol=1e-5
            )
    assert_allclose(
            sobel(I, axis=1, mode='constant')/4,
            L.gradv(I),
            rtol=0, atol=1e-5
            )

def test_variance():
    I = read_image('test/I.png')
    assert_allclose(I.var(), L.variance(I))

def test_solve():
    A = np.random.randn(7, 7).astype(np.float32)
    b = np.random.randn(7).astype(np.float32)

    assert_allclose(
            np.linalg.solve(A, b),
            L.solve(7, A, b)
            )

    assert_allclose(
            np.linalg.solve(A[:6,:6], b[:6]),
            L.solve(6, A, b)[:6]
            )

def test_BFGS_update():
    H = np.random.randn(7, 7).astype(np.float32)
    H = (H + H.T)/2
    s = np.random.randn(7).astype(np.float32)
    y = np.random.randn(7).astype(np.float32)

    H1 = (np.eye(6)-np.outer(s[:6], y[:6])/y[:6].dot(s[:6])).dot(H[:6,:6]).dot(np.eye(6)-np.outer(y[:6], s[:6])/y[:6].dot(s[:6])) + np.outer(s[:6], s[:6])/y[:6].dot(s[:6])
    L.BFGS_update(6, H, y, s)
    assert_allclose(H1, H[:6,:6])
