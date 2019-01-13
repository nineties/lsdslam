import time
from collections import namedtuple
import numpy as np
import scipy
import scipy.signal
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, zoom
from scipy.optimize import least_squares, minimize

# In the following codes we use following notations:
#
# - F_x:    (partial) differential coefficient d(F)/d(x).
# - F:      F in new frame
# - Fref:   F in reference frame
#
# and use following symbols:
#
# - Frame
#   - I: gray-scale camera image
#   - D: inverse depth image
#   - V: variance image
# - Points
#   - p: 2d point (u,v) in camera plane
#   - x: 3d point in camera coordinate system
# - Translation from reference frame to new frame
#   - n: rotation axis vector
#   - theta: rotation angle (theta = ||n||)
#   - R: rotation matrix
#   - t: translation vector
#   - rho: scaling factor (s=exp(rho))
#   - xi: combined vector [t,n] or [t,n,rho]
# - Maps
#   - tau: xref -> x

def zeros(shape): return np.zeros(shape, dtype=np.float32)
def ones(shape):  return np.ones(shape, dtype=np.float32)
def array(args):  return np.array(args, dtype=np.float32)
def eye(n):       return np.eye(n, dtype=np.float32)

def affine(coef, x):
    return coef[0].dot(x) + coef[1]

def compute_sR(s, n):
    "compute rotation matrix with scaling and its gradient"
    theta = np.linalg.norm(n)
    sR_n = zeros((3,3,3))
    if np.abs(theta) < 1e-30:
        # Avoid division by zero
        sR_n[0,1,2] = sR_n[1,2,0] = sR_n[2,0,1] = -s
        sR_n[0,2,1] = sR_n[1,0,2] = sR_n[2,1,0] = s
        return eye(3), sR_n

    sin = np.sin(theta)
    cos = np.cos(theta)
    c1 = s * (theta * cos - sin) / theta**3
    c2 = s * (theta * sin + 2 *cos - 2) / theta**4
    c3 = s * sin / theta
    c4 = s * (1 - cos) / theta**2

    N = array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    sR = s * eye(3) + c3 * N + c4 * N**2

    A = c1*N
    B = c2*N**2
    c40 = c4*n[0]
    c41 = c4*n[1]
    c42 = c4*n[2]
    C = np.array([[
        [   0,    c41,    c42],
        [ c41, -2*c40,    -c3],
        [ c42,     c3, -2*c40]
        ],[
        [-2*c41,  c40,     c3],
        [   c40,    0,    c42],
        [   -c3,  c42, -2*c41]
        ],[
        [-2*c42,    -c3,  c40],
        [    c3, -2*c42,  c41],
        [   c40,    c41,    0]
        ]])

    sR_n = n.reshape(3,1,1)*(A+B).reshape(1,3,3) + C
    return sR, sR_n

def compute_I(frame, size):
    "compute smoothed image, its gradient and variance"
    w, h, scale_w, scale_h = size
    I = zoom(frame, (scale_h, scale_w)).astype(np.float32)
    I_u = sobel(I, 0, mode='constant')/4
    I_v = sobel(I, 1, mode='constant')/4
    return I, I_u, I_v, I.var()

class Solver(object):
    def __init__(
            self,
            size,
            mask_thresh,
            huber_delta,
            eps,
            D0,
            V0,
            K
            ):
        self.size = size
        self.mask_thresh = mask_thresh
        self.huber_delta = huber_delta
        self.eps = eps
        self.D0 = D0
        self.V0 = V0

        # image size
        self.width  = None
        self.height = None

        # keyframe
        self.Iref = None
        self.Vref = None
        self.xref = None
        self.xref_D = None

        # current frame
        self.I = None
        self.I_u = None
        self.I_v = None
        self.Ivar = None

        # camera matrix
        self.K = K
        self.Kinv = np.linalg.inv(K)

        self.weighted_rp_memo = None

    def select_points(self, I, Iu, Iv):
        return np.where(Iu**2 + Iv**2 > self.mask_thresh**2)

    def set_initial_frame(self, frame):
        I, Iu, Iv, _ = compute_I(frame, self.size)
        points = self.select_points(I, Iu, Iv)
        n = len(points[0])
        pref = array(points)
        self.Iref = I[points]
        self.Dref = ones(n) * self.D0
        self.Vref = ones(n) * self.V0

        d = self.Dref.reshape(1,-1)

        self.xref = np.r_[pref/d, 1/d]           # pi^-1(pref, Dref)
        self.xref_D = np.r_[-pref/d**2, -1/d**2] # d(pi^-1)/d(Dref)(pref,Dref)

    def set_frame(self, frame):
        self.I, self.I_u, self.I_v, self.Ivar = compute_I(frame, self.size)

    def compute_residual(self, xi):
        # Memo result for jacobian
        if self.weighted_rp_memo and self.weighted_rp_memo[0] is xi:
            return self.weighted_rp_memo[1:]

        # degree of freedom
        dof = len(xi)

        s = 1 if dof == 6 else np.exp(xi[6])
        # Compute coefficients for translation from xref to x
        sR, sR_n  = compute_sR(s, xi[3:6])
        sKRKinv   = self.K.dot(sR).dot(self.Kinv)
        sKR_nKinv = zeros((3,3,3))
        sKR_nKinv[0] = self.K.dot(sR_n[0]).dot(self.Kinv)
        sKR_nKinv[1] = self.K.dot(sR_n[1]).dot(self.Kinv)
        sKR_nKinv[2] = self.K.dot(sR_n[2]).dot(self.Kinv)
        Kt = self.K.dot(xi[:3]).reshape(3,1)

        # translate points in reference frame to current frame
        x = sKRKinv.dot(self.xref) + Kt

        # project to camera plane
        p = x[:2]/x[2]

        py = p[0].astype(int)
        px = p[1].astype(int)

        # mask out points outside frame
        H, W = self.I.shape
        mask = (py<0)|(py>=H)|(px<0)|(px>=W)
        px[mask] = 0
        py[mask] = 0

        # photometric residual
        rp = self.Iref - self.I[py,px]

        # weight 
        I_u_x2 = -self.I_u[py,px]/x[2]
        I_v_x2 = -self.I_v[py,px]/x[2]

        I_x = np.vstack([I_u_x2, I_v_x2, I_u_x2*p[0] + I_v_x2*p[1]])
        tau_D = sKRKinv.dot(self.xref_D)
        I_D = np.einsum('ij,ij->j', I_x, tau_D)

        N = len(mask) - mask.sum()

        rp = (((2*self.Ivar + I_D**2 * self.Vref)**-0.5)/N) * rp
        rp[mask] = 0

        rows = [
            -I_x,
            -np.einsum('ij,nik,kj->nj', I_x, sKR_nKinv, self.xref)
            ]
        if dof==7:
            rows.append(
                    -(I_x*sKRKinv.dot(self.xref)).sum(0).reshape(1, -1)
                    )

        J = np.vstack(rows).T/N
        J[mask] = 0
        self.weighted_rp_memo = xi, rp, J

        return rp, J

    def residual(self, xi):
        return self.compute_residual(xi)[0]

    def residual_jac(self, xi):
        return self.compute_residual(xi)[1]

    def estimate_pose(self, xi):
        result = least_squares(
                fun=self.residual,
                jac=self.residual_jac,
                x0=xi,
                loss='huber',
                f_scale=self.huber_delta,
                xtol=self.eps,
                ftol=self.eps,
                gtol=self.eps
                )
        return result.x

class Tracker(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def init(self,
            D0=1.0,
            V0=1e5,
            mask_thresh=50.0,
            huber_delta=3,
            K=np.eye(3, dtype=np.float32),
            eps=0.001,
            pyramids=[(15,10), (30,20), (60,40), (120,80), (240,160)]
            ):
        self.frame = 0

        self.pyramids = []
        for size in pyramids:
            W, H = pyramids[-1]
            w, h = size
            self.pyramids.append((w, h, float(w)/W, float(h)/H))

        self.solvers = []
        for size in self.pyramids:
            self.solvers.append(Solver(
                size=size,
                mask_thresh=mask_thresh,
                huber_delta=huber_delta,
                eps=eps,
                D0=D0,
                V0=V0,
                K=K
                ))

    def set_initial_frame(self, frame):
        for solver in self.solvers:
            solver.set_initial_frame(frame)

    def set_frame(self, frame):
        for solver in self.solvers:
            solver.set_frame(frame)

    def estimate(self, frame):
        self.frame += 1
        if self.frame == 1:
            self.set_initial_frame(frame)
            return zeros(3), zeros(3)
        self.set_frame(frame)

        xi = zeros(6)
        for solver in self.solvers:
            xi = solver.estimate_pose(xi)
        return xi[:3], xi[3:]
