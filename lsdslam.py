import time
from collections import namedtuple
import numpy as np
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter, sobel
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
#   - p: camera plane (u,v)
#   - x: world coordinate
# - Translation from reference frame to new frame
#   - n: rotation axis vector
#   - theta: rotation angle (theta = ||n||)
#   - R: rotation matrix
#   - t: translation vector
#   - rho: scaling factor (s=exp(rho))
#   - xi: combined vector [t,n] or [t,n,rho]

def zeros(shape): return np.zeros(shape, dtype=np.float32)
def ones(shape):  return np.ones(shape, dtype=np.float32)
def array(args):  return np.array(args, dtype=np.float32)
def eye(n):       return np.eye(n, dtype=np.float32)

def multi_inner(A, B):
    """compute inner product of multiple vectors.

    let A=(a1,a2,...,an), B=(b1,b2,...,bn) then
    multi_inner(A, B) = (a1.b1, a2.b2, ..., an.bn)
    """
    return np.einsum('ij,ij->j', A, B)

def compute_R(n):
    "compute rotation matrix and its gradient"
    theta = np.linalg.norm(n)
    R_n = zeros((3,3,3))
    if np.abs(theta) < 1e-30:
        # Avoid division by zero
        R_n[0,1,2] = R_n[1,2,0] = R_n[2,0,1] = -1
        R_n[0,2,1] = R_n[1,0,2] = R_n[2,1,0] = 1
        return eye(3), R_n

    sin = np.sin(theta)
    cos = np.cos(theta)
    c1 = (theta * cos - sin) / theta**3
    c2 = (theta * sin + 2 *cos - 2) / theta**4
    c3 = sin / theta
    c4 = (1 - cos) / theta**2

    N = array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = eye(3) + c3 * N + c4 * N**2

    R_n[0] = c1*n[0]*N + c2*n[0]*N**2
    R_n[1] = c1*n[1]*N + c2*n[1]*N**2
    R_n[2] = c1*n[2]*N + c2*n[2]*N**2

    R_n[0,0,1] += c4*n[1]
    R_n[0,0,2] += c4*n[2]
    R_n[0,1,0] += c4*n[1]
    R_n[0,1,1] -= 2*c4*n[0]
    R_n[0,1,2] -= c3
    R_n[0,2,0] += c4*n[2]
    R_n[0,2,1] += c3
    R_n[0,2,2] -= 2*c4*n[0]

    R_n[1,0,0] -= 2*c4*n[1]
    R_n[1,0,1] += c4*n[0]
    R_n[1,0,2] += c3
    R_n[1,1,0] += c4*n[0]
    R_n[1,1,2] += c4*n[2]
    R_n[1,2,0] -= c3
    R_n[1,2,1] += c4*n[2]
    R_n[1,2,2] -= 2*c4*n[1]

    R_n[2,0,0] -= 2*c4*n[2]
    R_n[2,0,1] -= c3
    R_n[2,0,2] += c4*n[0]
    R_n[2,1,0] += c3
    R_n[2,1,1] -= 2*c4*n[2]
    R_n[2,1,2] += c4*n[1]
    R_n[2,2,0] += c4*n[0]
    R_n[2,2,1] += c4*n[1]

    return R, R_n

def compute_I(frame):
    "compute smoothed image, its gradient and variance"
    I = gaussian_filter(frame.astype(np.float32), 3, mode='constant')
    I_u = sobel(I, 0, mode='constant')/4
    I_v = sobel(I, 1, mode='constant')/4
    return I, I_u, I_v, I.var()

class Solver(object):
    def __init__(self):
        # keyframe
        self.pref = None
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
        self.K = None
        self.Kinv = None

        self.weighted_rp_memo = None

    def set_K(self, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def set_keyframe(self, pref, Iref, Dref, Vref):
        self.Iref = Iref
        self.Vref = Vref
        d = Dref.reshape(1,-1)

        self.xref = np.r_[pref/d, 1/d]           # pi^-1(pref, Dref)
        self.xref_D = np.r_[-pref/d**2, -1/d**2] # d(pi^-1)/d(Dref)(pref,Dref)

    def set_frame(self, frame):
        self.I, self.I_u, self.I_v, self.Ivar = compute_I(frame)

    def photometric_residual(self, xi, group):
        # Memo result for jacobian
        if self.weighted_rp_memo and self.weighted_rp_memo[0] is xi:
            return self.weighted_rp_memo[1:]

        # Compute warp
        s = 1 if group == 'SE3' else np.exp(xi[6])
        R, R_n = compute_R(xi[3:6])
        sKRKinv = s*self.K.dot(R).dot(self.Kinv)
        Kt = self.K.dot(xi[:3]).reshape(3, 1)
        sKR_n0Kinv = s*self.K.dot(R_n[0]).dot(self.Kinv)
        sKR_n1Kinv = s*self.K.dot(R_n[1]).dot(self.Kinv)
        sKR_n2Kinv = s*self.K.dot(R_n[2]).dot(self.Kinv)

        # translate points in reference frame to current frame
        y = sKRKinv.dot(self.xref) + Kt

        # project to camera plane
        q = y[:2]/y[2]

        p = q.astype(int)

        mask = ~((p[0] >= 0)&(p[0] < self.I.shape[0])&(p[1] >= 0)&(p[1] < self.I.shape[1]))
        p[:, mask] = 0

        # residual
        r = self.Iref - self.I[p[0],p[1]]

        # weight 
        I_u_y2 = self.I_u[p[0],p[1]]/y[2]
        I_v_y2 = self.I_v[p[0],p[1]]/y[2]

        I_y = np.vstack([-I_u_y2, -I_v_y2, -I_u_y2*q[0] - I_v_y2*q[1]])
        tau_D = sKRKinv.dot(self.xref_D)

        I_D = multi_inner(I_y, tau_D)
        #I_D = (I_y * tau_D).sum(0)

        N = len(mask) - mask.sum()

        rp = ((2*self.Ivar + I_D**2 * self.Vref)**-0.5 * r)/N
        rp[mask] = 0

        rows = [
            -I_y,
            -(I_y*sKR_n0Kinv.dot(self.xref)).sum(0).reshape(1, -1),
            -(I_y*sKR_n1Kinv.dot(self.xref)).sum(0).reshape(1, -1),
            -(I_y*sKR_n2Kinv.dot(self.xref)).sum(0).reshape(1, -1),
            ]
        if group == 'Sim3':
            rows.append(
                    -(I_y*sKRKinv.dot(self.xref)).sum(0).reshape(1, -1)
                    )

        N = len(mask) - mask.sum()
        J = np.vstack(rows).T/N
        J[mask] = 0
        self.weighted_rp_memo = xi, rp, J

        return rp, J

    def weighted_rp(self, xi, group):
        return self.photometric_residual(xi, group)[0]

    def weighted_rp_jac(self, xi, group):
        return self.photometric_residual(xi, group)[1]

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
            ):
        self.frame = 0

        self.mask_thresh = mask_thresh
        self.huber_delta = huber_delta
        self.D0 = D0
        self.V0 = V0

        self.eps = eps
        self.solver = Solver()
        self.solver.set_K(K)

    def set_initial_frame(self, frame):
        I, gu, gv, _ = compute_I(frame)
        points = np.where(np.sqrt(gu**2 + gv**2) > self.mask_thresh)
        n = len(points[0])
        self.solver.set_keyframe(
                pref=array(points),
                Iref=I[points],
                Dref=ones(n) * self.D0,
                Vref=ones(n) * self.V0
                )

    def estimate(self, I):
        self.frame += 1
        if self.frame == 1:
            self.set_initial_frame(I)
            return zeros(3), zeros(3)
        self.solver.set_frame(I)

        start = time.time()
        result = least_squares(
                fun=self.solver.weighted_rp,
                jac=self.solver.weighted_rp_jac,
                x0=zeros(6),
                loss='huber',
                f_scale=self.huber_delta,
                xtol=self.eps,
                ftol=self.eps,
                gtol=self.eps,
                args=('SE3',)
                )
        elapsed = time.time() - start
        print(result)
        print('{}ms'.format(elapsed*1000))
        t = result.x[:3]
        n = result.x[3:6]
        return t, n

