import time
import numpy as np
from sympy import Matrix, symbols, diff, simplify, lambdify, transpose, cos, sin, exp, eye
from scipy.signal import convolve2d
from scipy.ndimage import sobel
from PIL import Image

import lsdslam.lib as L

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

def read_image(path):
    return np.asarray(Image.open(path).convert('P').resize(L.SIZE), dtype=np.float32)

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
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
            R(n, theta),
            L.R(n, theta)
            )

def test_R_theta():
    n = random_norm(3)
    theta = np.random.randn()

    assert_allclose(
            R_theta(n, theta),
            L.R_theta(n, theta)
            )

def test_R_n():
    n = random_norm(3)
    theta = np.random.randn()

    A = L.R_n(n, theta)

    for i in range(3):
        assert_allclose(R_n[i](n, theta), A[i])

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

def test_KTKinv():
    x = random_vec(3)
    n = random_norm(3)
    t = random_vec(3)
    theta = np.random.randn()
    rho = np.random.randn()
    K = np.random.randn(3, 3).astype(np.float32)

    A, b = L.precompute_KTKinv(K, rho, n, theta, t)
    assert_allclose(
            K.dot(T(rho, n, theta, t, np.linalg.inv(K).dot(x)).flatten()),
            A.dot(x) + b
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

def test_rp():
    I = read_image('test/I.png')
    Iref = read_image('test/Iref.png')
    Dref = read_image('test/Dref.png')
    Vref = np.ones_like(Dref)

    t = random_norm(3)*0.01
    n = random_norm(3)
    theta = 0.01
    rho = 1.1
    K = (np.eye(3) + np.random.randn(3, 3)*1e-5).astype(np.float32)

    p_ref = (50, 50)

    x_ref = np.linalg.inv(K).dot(piinv(p_ref, Dref[p_ref])).flatten()
    x = T(rho, n, theta, t, x_ref).flatten()
    u, v = pip(K.dot(x)).flatten()
    if u < 0 or u >= I.shape[0] or v < 0 or v >= I.shape[1]:
        rp1 = 0
    else:
        rp1 = Iref[p_ref] - I[int(u), int(v)]

    slam = L.LSDSLAMStruct()
    L.precompute_cache(
        slam,
        Iref, Dref, Vref, I,
        K, rho, n, theta, t
        )
    rp2 = L.rp(slam, p_ref)

    assert_allclose(rp1, rp2)
