import time
from collections import namedtuple
import numpy as np
import scipy
import scipy.signal
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import least_squares, minimize

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def ones(shape):
    return np.ones(shape, dtype=np.float32)

def eye(n):
    return np.eye(n, dtype=np.float32)

def array(args):
    return np.array(args, dtype=np.float32)

def preprocess_frame(I):
    I = gaussian_filter(I, 3, mode='constant')
    Iu = sobel(I, 0, mode='constant')/4 # dI/du
    Iv = sobel(I, 1, mode='constant')/4 # dI/dv
    return I, Iu, Iv

# Rotation matrix
def compute_R(n):
    theta = np.linalg.norm(n)
    R_n0 = zeros((3,3))
    R_n1 = zeros((3,3))
    R_n2 = zeros((3,3))
    if np.abs(theta) < 1e-30:
        R_n0[1,2] = -1
        R_n0[2,1] = 1
        R_n1[0,2] = 1
        R_n1[2,0] = -1
        R_n2[0,1] = -1
        R_n2[1,0] = 1
        return eye(3), R_n0, R_n1, R_n2

    sin = np.sin(theta)
    cos = np.cos(theta)
    c1 = (theta * cos - sin) / theta**3
    c2 = (theta * sin + 2 *cos - 2) / theta**4
    c3 = sin / theta
    c4 = (1 - cos) / theta**2

    N = array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    N2 = N**2

    R = eye(3) + c3 * N + c4 * N**2

    R_n0[0,0] = c1*n[0]*N[0,0] + c2*n[0]*N2[0,0]
    R_n0[0,1] = c1*n[0]*N[0,1] + c2*n[0]*N2[0,1] + c4*n[1]
    R_n0[0,2] = c1*n[0]*N[0,2] + c2*n[0]*N2[0,2] + c4*n[2]
    R_n0[1,0] = c1*n[0]*N[1,0] + c2*n[0]*N2[1,0] + c4*n[1]
    R_n0[1,1] = c1*n[0]*N[1,1] + c2*n[0]*N2[1,1] - 2*c4*n[0]
    R_n0[1,2] = c1*n[0]*N[1,2] + c2*n[0]*N2[1,2] - c3
    R_n0[2,0] = c1*n[0]*N[2,0] + c2*n[0]*N2[2,0] + c4*n[2]
    R_n0[2,1] = c1*n[0]*N[2,1] + c2*n[0]*N2[2,1] + c3
    R_n0[2,2] = c1*n[0]*N[2,2] + c2*n[0]*N2[2,2] - 2*c4*n[0]

    R_n1[0,0] = c1*n[1]*N[0,0] + c2*n[1]*N2[0,0] - 2*c4*n[1]
    R_n1[0,1] = c1*n[1]*N[0,1] + c2*n[1]*N2[0,1] + c4*n[0]
    R_n1[0,2] = c1*n[1]*N[0,2] + c2*n[1]*N2[0,2] + c3
    R_n1[1,0] = c1*n[1]*N[1,0] + c2*n[1]*N2[1,0] + c4*n[0]
    R_n1[1,1] = c1*n[1]*N[1,1] + c2*n[1]*N2[1,1]
    R_n1[1,2] = c1*n[1]*N[1,2] + c2*n[1]*N2[1,2] + c4*n[2]
    R_n1[2,0] = c1*n[1]*N[2,0] + c2*n[1]*N2[2,0] - c3
    R_n1[2,1] = c1*n[1]*N[2,1] + c2*n[1]*N2[2,1] + c4*n[2]
    R_n1[2,2] = c1*n[1]*N[2,2] + c2*n[1]*N2[2,2] - 2*c4*n[1]

    R_n2[0,0] = c1*n[2]*N[0,0] + c2*n[2]*N2[0,0] - 2*c4*n[2]
    R_n2[0,1] = c1*n[2]*N[0,1] + c2*n[2]*N2[0,1] - c3
    R_n2[0,2] = c1*n[2]*N[0,2] + c2*n[2]*N2[0,2] + c4*n[0]
    R_n2[1,0] = c1*n[2]*N[1,0] + c2*n[2]*N2[1,0] + c3
    R_n2[1,1] = c1*n[2]*N[1,1] + c2*n[2]*N2[1,1] - 2*c4*n[2]
    R_n2[1,2] = c1*n[2]*N[1,2] + c2*n[2]*N2[1,2] + c4*n[1]
    R_n2[2,0] = c1*n[2]*N[2,0] + c2*n[2]*N2[2,0] + c4*n[0]
    R_n2[2,1] = c1*n[2]*N[2,1] + c2*n[2]*N2[2,1] + c4*n[1]
    R_n2[2,2] = c1*n[2]*N[2,2] + c2*n[2]*N2[2,2]

    return R, R_n0, R_n1, R_n2

Keyframe = namedtuple('Keyframe', 'p I D V')

class Solver(object):
    def __init__(self):
        # keyframe
        self.ref = None
        self.piinv = None
        self.piinv_D = None

        # current frame
        self.I = None
        self.Iu = None
        self.Iv = None
        self.Ivar = None

        # camera matrix
        self.K = None
        self.Kinv = None

        self.weighted_rp_memo = None

    def set_K(self, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def set_keyframe(self, keyframe):
        self.ref = keyframe

        x = keyframe.p
        d = keyframe.D
        self.piinv = array([x[0]/d, x[1]/d, 1/d])               # pi^-1(p, D)
        self.piinv_D = array([-x[0]/d**2, -x[1]/d**2, -1/d**2]) # d(pi^-1)/d(D)(p,D)

    def set_frame(self, I):
        self.I, self.Iu, self.Iv = preprocess_frame(I)
        self.Ivar = self.I.var()

    def photometric_residual(self, xi, space):
        # Memo result for jacobian
        if self.weighted_rp_memo and self.weighted_rp_memo[0] is xi:
            return self.weighted_rp_memo[1:]

        # Compute warp
        s = 1 if space == 'SE3' else np.exp(xi[6])
        R, R_n0, R_n1, R_n2 = compute_R(xi[3:6])
        sKRKinv = s*self.K.dot(R).dot(self.Kinv)
        Kt = self.K.dot(xi[:3]).reshape(3, 1)
        sKR_n0Kinv = s*self.K.dot(R_n0).dot(self.Kinv)
        sKR_n1Kinv = s*self.K.dot(R_n1).dot(self.Kinv)
        sKR_n2Kinv = s*self.K.dot(R_n2).dot(self.Kinv)

        # translate points in reference frame to current frame
        y = sKRKinv.dot(self.piinv) + Kt

        # project to camera plane
        q = y[:2]/y[2]

        p = q.astype(int)

        mask = ~((p[0] >= 0)&(p[0] < self.I.shape[0])&(p[1] >= 0)&(p[1] < self.I.shape[1]))
        p[:, mask] = 0

        # residual
        r = self.ref.I - self.I[p[0],p[1]]

        # weight 
        Iu_y2 = self.Iu[p[0],p[1]]/y[2]
        Iv_y2 = self.Iv[p[0],p[1]]/y[2]

        I_y = np.vstack([-Iu_y2, -Iv_y2, -Iu_y2*q[0] - Iv_y2*q[1]])
        tau_D = sKRKinv.dot(self.piinv_D)

        I_D = (I_y * tau_D).sum(0)

        N = len(mask) - mask.sum()

        rp = ((2*self.Ivar + I_D**2 * self.ref.V)**-0.5 * r)/N
        rp[mask] = 0


        # 3xn
        rows = [
            -I_y,
            -(I_y*sKR_n0Kinv.dot(self.piinv)).sum(0).reshape(1, -1),
            -(I_y*sKR_n1Kinv.dot(self.piinv)).sum(0).reshape(1, -1),
            -(I_y*sKR_n2Kinv.dot(self.piinv)).sum(0).reshape(1, -1),
            ]
        if space == 'Sim3':
            rows.append(
                    -(I_y*sKRKinv.dot(self.piinv)).sum(0).reshape(1, -1)
                    )

        N = len(mask) - mask.sum()
        J = np.vstack(rows).T/N
        J[mask] = 0
        self.weighted_rp_memo = xi, rp, J

        return rp, J

    def weighted_rp(self, xi, space):
        return self.photometric_residual(xi, space)[0]

    def weighted_rp_jac(self, xi, space):
        return self.photometric_residual(xi, space)[1]

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

    def set_initial_frame(self, I):
        I, gu, gv = preprocess_frame(I)
        points = np.where(np.sqrt(gu**2 + gv**2) > self.mask_thresh)
        n = len(points[0])
        ref = Keyframe(
                p=array(points),
                I=I[points],
                D=ones(n) * self.D0,
                V=ones(n) * self.V0
                )
        self.solver.set_keyframe(ref)

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

