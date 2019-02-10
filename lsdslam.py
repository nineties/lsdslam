import time
from collections import namedtuple
import numpy as np
import scipy
import scipy.signal
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, zoom
from scipy.optimize import least_squares, minimize
from threading import Thread

# In the following codes we use following notations:
#
# - F_x: (partial) differential coefficient d(F)/d(x).
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
#   - tau: ref.x -> x

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


class Frame(object):
    def __init__(self, frame, size):
        "compute smoothed image, its gradient and variance"
        w, h, dx, dy = size
        self.size = size
        self.I = zoom(frame, (1/dy, 1/dx)).astype(np.float32)

        self.I_u = sobel(self.I, 0, mode='constant')/(4*dy)
        self.I_v = sobel(self.I, 1, mode='constant')/(4*dx)
        self.Ivar = self.I.var()

class KeyFrame(Frame):
    def __init__(self, frame, mask_thresh, D0, V0):
        w, h, dx, dy = frame.size
        points = np.where(frame.I_u**2 + frame.I_v**2 > mask_thresh**2)
        n = len(points[0])

        self.frame = frame
        self.I = frame.I[points]
        self.D = ones(n) * D0
        self.V = ones(n) * V0

        p = array(points)
        p[0] *= dy
        p[1] *= dx
        d = self.D.reshape(1,-1)

        self.x = np.r_[p/d, 1/d]           # pi^-1(pref, Dref)
        self.x_D = np.r_[-p/d**2, -1/d**2] # d(pi^-1)/d(Dref)(pref,Dref)
        self.meanD = 1


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

        # camera matrix
        self.K = K
        self.Kinv = np.linalg.inv(K)

        self.weighted_rp_memo = None

    def set_initial_frame(self, frame):
        self.ref = KeyFrame(Frame(frame, self.size), self.mask_thresh,
                self.D0, self.V0)

    def set_frame(self, frame):
        self.frame = Frame(frame, self.size)

    def compute_residual(self, xi):
        # Memo result for jacobian
        if self.weighted_rp_memo and self.weighted_rp_memo[0] is xi:
            return self.weighted_rp_memo[1:]

        # degree of freedom
        dof = len(xi)

        s = 1 if dof == 6 else np.exp(xi[6])
        # Compute coefficients for translation from ref.x to x
        sR, sR_n  = compute_sR(s, xi[3:6])
        sKRKinv   = self.K.dot(sR).dot(self.Kinv)
        sKR_nKinv = zeros((3,3,3))
        sKR_nKinv[0] = self.K.dot(sR_n[0]).dot(self.Kinv)
        sKR_nKinv[1] = self.K.dot(sR_n[1]).dot(self.Kinv)
        sKR_nKinv[2] = self.K.dot(sR_n[2]).dot(self.Kinv)
        Kt = self.K.dot(xi[:3]).reshape(3,1)

        # translate points in reference frame to current frame
        x = sKRKinv.dot(self.ref.x) + Kt

        # project to camera plane
        p = x[:2]/x[2]

        py = p[0].astype(int)
        px = p[1].astype(int)

        # mask out points outside frame
        H, W = self.frame.I.shape
        mask = (py<0)|(py>=H)|(px<0)|(px>=W)
        px[mask] = 0
        py[mask] = 0

        # photometric residual
        rp = self.ref.I - self.frame.I[py,px]

        # weight 
        I_u_x2 = -self.frame.I_u[py,px]/x[2]
        I_v_x2 = -self.frame.I_v[py,px]/x[2]

        I_x = np.vstack([I_u_x2, I_v_x2, I_u_x2*p[0] + I_v_x2*p[1]])
        tau_D = sKRKinv.dot(self.ref.x_D)
        I_D = np.einsum('ij,ij->j', I_x, tau_D)

        N = len(mask) - mask.sum()

        rp = (((2*self.frame.Ivar + I_D**2 * self.ref.V)**-0.5)/N) * rp
        rp[mask] = 0

        rows = [
            -I_x,
            -np.einsum('ij,nik,kj->nj', I_x, sKR_nKinv, self.ref.x)
            ]
        if dof==7:
            rows.append(
                    -(I_x*sKRKinv.dot(self.ref.x)).sum(0).reshape(1, -1)
                    )

        J = np.vstack(rows).T/N
        J[mask] = 0
        self.weighted_rp_memo = xi, rp, J
        self.point_usage = (mask == 0).mean()

        return rp, J

    def residual(self, xi):
        return self.compute_residual(xi)[0]

    def residual_jac(self, xi):
        return self.compute_residual(xi)[1]

    def estimate_pose(self, xi):

        # skip estimation when reference points are empty
        if self.ref.x.shape[1] == 0:
            return xi

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
    def __init__(self,
            D0=1.0,
            V0=1e5,
            mask_thresh=50.0,
            huber_delta=3,
            K=np.eye(3, dtype=np.float32),
            eps=0.001,
            pyramids=[(30,20), (60,40), (120,80), (240,160)]
            ):
        self.frame = 0

        self.pyramids = []
        for w, h in pyramids:
            W, H = pyramids[-1]     # resolution of level 0
            self.pyramids.append((w, h, W/float(w), H/float(h)))

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

    def get_keyframe(self):
        return self.solvers[-1].ref

    def estimate(self, frame, t, n, rho=None):
        if rho is None:
            return self.estaimte_se3(frame, t, n)
        else:
            return self.estimate_sim3(frame, t, n, rho)

    def estaimte_se3(self, frame, t, n):
        self.frame += 1
        if self.frame == 1:
            self.set_initial_frame(frame)
            return zeros(3), zeros(3), 0
        self.set_frame(frame)

        xi = np.r_[t, n]
        for solver in self.solvers:
            xi = solver.estimate_pose(xi)
        return xi[:3], xi[3:], self.solvers[-1].point_usage

    def estimate_sim3(self, frame, t, n, rho):
        self.frame += 1
        if self.frame == 1:
            self.set_initial_frame(frame)
            return zeros(3), zeros(3), 0, 0
        self.set_frame(frame)

        xi = np.r_[t, n, rho]
        for solver in self.solvers:
            xi = solver.estimate_pose(xi)
        return xi[:3], xi[3:6], xi[6], self.solvers[-1].point_usage

class PoseGraph(object):
    def __init__(self):
        self.poses = []

    def add_pose(self, t, n, rho=0):
        self.poses.append((t, n, rho))

class MappingThread(Thread):
    def __init__(self, system):
        super().__init__()
        self.system = system

    def run(self):
        while not self.system.terminate:
            print('mapping')

class ConstraintThread(Thread):
    def __init__(self, system):
        super().__init__()
        self.system = system

    def run(self):
        while not self.system.terminate:
            print('constraint')

class OptimizationThread(Thread):
    def __init__(self, system):
        super().__init__()
        self.system = system

    def run(self):
        while not self.system.terminate:
            print('optimization')

class SLAMSystem(object):
    def __init__(self,
            dist_weight=16,
            usage_weight=9,
            init_phase_count=5,
            ):
        self.init_phase_count = init_phase_count
        self.dist_weight = dist_weight
        self.usage_weight = usage_weight

        # Pose
        self.t = zeros(3)   # transition
        self.n = zeros(3)   # rotation axis
        self.rho = 0        # log(rotation angle)

        self.tracker = Tracker()
        self.graph = PoseGraph()

        self.terminate = False
        self.mapping_thread = MappingThread(self)
        self.constraint_thread = ConstraintThread(self)
        self.optimization_thread = OptimizationThread(self)

    def __enter__(self):
        self.mapping_thread.start()
        self.constraint_thread.start()
        self.optimization_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate = True
        self.mapping_thread.join()
        self.constraint_thread.join()
        self.optimization_thread.join()

    def track(self, frame):
        # Estimate current pose
        t, n, usage = self.tracker.estimate(frame, self.t, self.n)
        self.graph.add_pose(t, n)

        # Select keyframe
        dist = t * self.tracker.get_keyframe().meanD
        score = self.dist_weight*dist.dot(dist) + self.usage_weight*(1-usage)**2

        if len(self.graph.poses) < self.init_phase_count:
            thresh = 0.14 + 0.56 * len(self.graph.poses) / self.init_phase_count
        else:
            thresh = 1
        if score > thresh:
            print('new key frame')
        else:
            print('no new key frame')
