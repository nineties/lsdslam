#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "lsdslam.h"

static inline float
square(float x)
{
    return x*x;
}

int
get_imagewidth(void)
{
    return WIDTH;
}

int
get_imageheight(void)
{
    return HEIGHT;
}

/* matrix-vector multiplication */
void
mul_NTNT(int l, int m, int n, float *c, float *a, float *b)
{
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            float v = 0.0;
            for (int k = 0; k < m; k++) {
                v += a[i*m+k]*b[k*n+j];
            }
            c[i*n+j] = v;
        }
    }
}

void
mul_NTT(int l, int m, int n, float *c, float *a, float *b)
{
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            float v = 0.0;
            for (int k = 0; k < m; k++) {
                v += a[i*m+k]*b[j*m+k];
            }
            c[i*n+j] = v;
        }
    }
}

void
mulmv3d(float y[3], float A[3][3], float x[3])
{
    for (int i = 0; i < 3; i++) {
        float v = 0.0;
        for (int k = 0; k < 3; k++) {
            v += A[i][k] * x[k];
        }
        y[i] = v;
    }
}

void
mul3x3(float y[3][3], float a[3][3], float b[3][3])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float v = 0.0;
            for (int k = 0; k < 3; k++) {
                v += a[i][k] * b[k][j];
            }
            y[i][j] = v;
        }
    }
}

void
mul3x3_twice(float y[3][3], float a[3][3], float b[3][3], float c[3][3])
{
    float t[3][3];
    mul3x3(t, b, c);
    mul3x3(y, a, t);
}

/* determinant of 3x3 matrix */
float
det3x3(float x[3][3])
{
    return x[0][0]*x[1][1]*x[2][2] -
           x[0][0]*x[1][2]*x[2][1] +
           x[0][1]*x[1][2]*x[2][0] -
           x[0][1]*x[1][0]*x[2][2] +
           x[0][2]*x[2][1]*x[1][0] -
           x[0][2]*x[2][0]*x[1][1];
}

/* inverse of 3x3 matrix */
void
inv3x3(float y[3][3], float x[3][3])
{
    float det = det3x3(x);
    y[0][0] = (x[1][1]*x[2][2] - x[1][2]*x[2][1])/det;
    y[0][1] = (x[0][2]*x[2][1] - x[0][1]*x[2][2])/det;
    y[0][2] = (x[0][1]*x[1][2] - x[0][2]*x[1][1])/det;
    y[1][0] = (x[1][2]*x[2][0] - x[1][0]*x[2][2])/det;
    y[1][1] = (x[0][0]*x[2][2] - x[0][2]*x[2][0])/det;
    y[1][2] = (x[0][2]*x[1][0] - x[0][0]*x[1][2])/det;
    y[2][0] = (x[1][0]*x[2][1] - x[1][1]*x[2][0])/det;
    y[2][1] = (x[0][1]*x[2][0] - x[0][0]*x[2][1])/det;
    y[2][2] = (x[0][0]*x[1][1] - x[0][1]*x[1][0])/det;
}

void
affine3d(float y[3], float A[3][3], float b[3], float x[3])
{
    mulmv3d(y, A, x);
    for (int i = 0; i < 3; i++)
        y[i] += b[i];
}

/**** Rotation ****/

/* Rodrigues's rotation formula
 * Rotate x according to axis n and angle theta.
 * The result is stored to y */
void
compute_R(float y[3][3], float n[3], float theta)
{
    float s = sinf(theta);
    float c = cosf(theta);

    y[0][0] = c + n[0]*n[0]*(1-c);
    y[0][1] = n[0]*n[1]*(1-c) - n[2]*s;
    y[0][2] = n[2]*n[0]*(1-c) + n[1]*s;
    y[1][0] = n[0]*n[1]*(1-c) + n[2]*s;
    y[1][1] = c + n[1]*n[1]*(1-c);
    y[1][2] = n[1]*n[2]*(1-c) - n[0]*s;
    y[2][0] = n[2]*n[0]*(1-c) - n[1]*s;
    y[2][1] = n[1]*n[2]*(1-c) + n[0]*s;
    y[2][2] = c + n[2]*n[2]*(1-c);
}

// dR/d(theta)
void
compute_R_theta(float y[3][3], float n[3], float theta)
{
    float s = sinf(theta);
    float c = cosf(theta);

    y[0][0] = -s + n[0]*n[0]*s;
    y[0][1] = n[0]*n[1]*s - n[2]*c;
    y[0][2] = n[2]*n[0]*s + n[1]*c;
    y[1][0] = n[0]*n[1]*s + n[2]*c;
    y[1][1] = -s + n[1]*n[1]*s;
    y[1][2] = n[1]*n[2]*s - n[0]*c;
    y[2][0] = n[2]*n[0]*s - n[1]*c;
    y[2][1] = n[1]*n[2]*s + n[0]*c;
    y[2][2] = -s + n[2]*n[2]*s;
}

// dR/dn
void
compute_R_n(float y[3][3][3], float n[3], float theta)
{
    float s = sinf(theta);
    float c = cosf(theta);

    // dR/dn1
    y[0][1][1] = y[0][2][2] = -2*(1-c)*n[0];
    y[0][0][1] = y[0][1][0] = (1-c)*n[1];
    y[0][0][2] = y[0][2][0] = (1-c)*n[2];
    y[0][1][2] = -s;
    y[0][2][1] = s;

    // dR/dn2
    y[1][0][1] = y[1][1][0] = (1-c)*n[0];
    y[1][1][2] = y[1][2][1] = (1-c)*n[2];
    y[1][0][0] = y[1][2][2] = -2*(1-c)*n[1];
    y[1][0][2] = s;
    y[1][2][0] = -s;

    // dR/dn2
    y[2][0][0] = y[2][1][1] = -2*(1-c)*n[2];
    y[2][0][2] = y[2][2][0] = (1-c)*n[0];
    y[2][1][2] = y[2][2][1] = (1-c)*n[1];
    y[2][0][1] = -s;
    y[2][1][0] = s;
}

/**** Similarity Transformation* ****/
/* Ax+b = sRx + t */
void
precompute_T(float A[3][3], float b[3], float rho, float n[3], float theta, float t[3])
{
    for (int i = 0; i < 3; i++)
        b[i] = t[i];

    compute_R(A, n, theta);
    float s = expf(rho);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j] *= s;
        }
    }
}

/**** Projection to camera plane ****/
void
pi(float y[3], float x[3])
{
    y[0] = x[0]/x[2];
    y[1] = x[1]/x[2];
    y[2] = 1/x[2];
}

void
pip_x(float y[2][3], float x[3])
{
    y[0][0] = 1/x[2];
    y[0][1] = 0;
    y[0][2] = -x[0]/(x[2]*x[2]);
    y[1][0] = 0;
    y[1][1] = 1/x[2];
    y[1][2] = -x[1]/(x[2]*x[2]);
}

void
piinv(float y[3], float x[2], float d)
{
    y[0] = x[0]/d;
    y[1] = x[1]/d;
    y[2] = 1/d;
}

void
piinv_d(float y[3], float x[2], float d)
{
    y[0] = -x[0]/(d*d);
    y[1] = -x[1]/(d*d);
    y[2] = -1/(d*d);
}

void
filter3x3(float y[HEIGHT][WIDTH], float k[3][3], float x[HEIGHT][WIDTH])
{
    float pad_x[HEIGHT+2][WIDTH+2] = {0};

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++)
            pad_x[i+1][j+1] = x[i][j];
    }

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float s = 0;
            for (int u = 0; u < 3; u++) {
                for (int v = 0; v < 3; v++) {
                    s += k[2-u][2-v]*pad_x[i+u][j+v];
                }
            }
            y[i][j] = s;
        }
    }
}

void
gaussian_filter3x3(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        {1/9., 1/9., 1/9.},
        {1/9., 1/9., 1/9.},
        {1/9., 1/9., 1/9.},
    };
    filter3x3(y, k, x);
}

void
gradu(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        { 1/4., 2/4., 1/4.},
        { 0., 0., 0.},
        {-1/4.,-2/4.,-1/4.},
    };
    filter3x3(y, k, x);
}

void
gradv(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        {1/4., 0., -1/4.},
        {2/4., 0., -2/4.},
        {1/4., 0., -1/4.},
    };
    filter3x3(y, k, x);
}

float
variance(float x[HEIGHT][WIDTH])
{
    float mean = 0.0;
    float sqmean = 0.0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            mean += x[i][j];
            sqmean += x[i][j]*x[i][j];
        }
    }
    mean /= (HEIGHT*WIDTH);
    sqmean /= (HEIGHT*WIDTH);
    return sqmean - mean*mean;
}

void
create_mask(bool mask[HEIGHT][WIDTH], float I[HEIGHT][WIDTH], float thresh)
{
    float gu[HEIGHT][WIDTH];
    float gv[HEIGHT][WIDTH];
    gradu(gu, I);
    gradv(gv, I);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++)
            mask[i][j] = sqrtf(square(gu[i][j]) + square(gv[i][j])) > thresh;
    }
}

void
precompute_cache(
        struct lsdslam *slam,
        float Iref[HEIGHT][WIDTH],
        float Dref[HEIGHT][WIDTH],
        float Vref[HEIGHT][WIDTH],
        float I[HEIGHT][WIDTH],
        float K[3][3],
        float rho,
        float n[3],
        float theta,
        float t[3]
        )
{
    struct compute_cache *cache = &slam->cache;
    create_mask(cache->mask, Iref, slam->param.mask_thresh);

    memcpy(cache->Iref, Iref, sizeof(cache->Iref));
    memcpy(cache->Dref, Dref, sizeof(cache->Dref));
    memcpy(cache->Vref, Vref, sizeof(cache->Vref));

    memcpy(cache->I, I, sizeof(cache->I));
    gradu(cache->I_u, I);
    gradv(cache->I_v, I);
    cache->Ivar = variance(I);

    float Kinv[3][3] = {0};
    inv3x3(Kinv, K);

    float sR[3][3] = {0};
    float s = expf(rho);
    compute_R(sR, n, theta);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sR[i][j] *= s;

    mul3x3_twice(cache->sKRKinv, K, sR, Kinv);
    mulmv3d(cache->Kt, K, t);

    float sR_n[3][3][3] = {0};
    compute_R_n(sR_n, n, theta);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                sR_n[i][j][k] *= s;
    mul3x3_twice(cache->sKR_nKinv[0], K, sR_n[0], Kinv);
    mul3x3_twice(cache->sKR_nKinv[1], K, sR_n[1], Kinv);
    mul3x3_twice(cache->sKR_nKinv[2], K, sR_n[2], Kinv);

    float sR_theta[3][3] = {0};
    compute_R_theta(sR_theta, n, theta);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sR_theta[i][j] *= s;
    mul3x3_twice(cache->sKR_thetaKinv, K, sR_theta, Kinv);
}

// Compute photometric residual, derivative wrt xi and weight.
// *res will be NaN for out-bound error.
void
photometric_residual(struct lsdslam *slam,
        float *res, float J[8], float *w,
        int u_ref, int v_ref)
{
    struct compute_cache *cache = &slam->cache;
    float p_ref[2] = {u_ref, v_ref};
    float x[3];
    float y[3];
    float q[3];

    piinv(x, p_ref, cache->Dref[u_ref][v_ref]);
    affine3d(y, cache->sKRKinv, cache->Kt, x);
    pi(q, y);

    int u = (int)q[0];
    int v = (int)q[1];

    if (u < 0 || u >= HEIGHT || v < 0 || v >= WIDTH) {
        *res = NAN;
        return;
    }

    *res = cache->Iref[u_ref][v_ref] - cache->I[u][v];

    /* d(pi_p)/dy */
    float pip_y[2][3];
    pip_x(pip_y, y);

    /* dI/dq */
    float I_q[2] = {cache->I_u[u][v], cache->I_v[u][v]};

    /* dI/dy */
    float I_y[3] = {
        I_q[0]*pip_y[0][0] + I_q[1]*pip_y[1][0],
        I_q[0]*pip_y[0][1] + I_q[1]*pip_y[1][1],
        I_q[0]*pip_y[0][2] + I_q[1]*pip_y[1][2],
    };

    /* ==== Compute d(r_p)/d(xi) ==== */

    /* transpose of d(tau)/d(xi) */
    float tau_xi_T[8][3] = {0};

    mulmv3d(tau_xi_T[0], cache->sKRKinv, x);       // d(tau)/d(rho)(x) = sKRK^-1x 
    mulmv3d(tau_xi_T[1], cache->sKR_nKinv[0], x);  // d(tau)/d(n_1)(x) = sKR_n_1K^-1x
    mulmv3d(tau_xi_T[2], cache->sKR_nKinv[1], x);  // d(tau)/d(n_2)(x) = sKR_n_2K^-1x
    mulmv3d(tau_xi_T[3], cache->sKR_nKinv[2], x);  // d(tau)/d(n_3)(x) = sKR_n_3K^-1x
    mulmv3d(tau_xi_T[4], cache->sKR_thetaKinv, x); // d(tau)/d(theta)(x) = sKR_thetaK^-1x
    // d(tau)/d(t)(x) = I
    tau_xi_T[5][0] = 1;
    tau_xi_T[5][1] = 0;
    tau_xi_T[5][2] = 0;
    tau_xi_T[6][0] = 0;
    tau_xi_T[6][1] = 1;
    tau_xi_T[6][2] = 0;
    tau_xi_T[7][0] = 0;
    tau_xi_T[7][1] = 0;
    tau_xi_T[7][2] = 1;

    // J = -dI/d(xi)
    mul_NTT(1, 3, 8, (float*)J, (float*)I_y, (float*)tau_xi_T);
    for (int i = 0; i < 8; i++)
        J[i] *= -1;

    /* ==== Compute d(r_p)d(D_ref) */

    /* dI/dx = dI/dy*d(tau)/dx
     * Note: d(tau)/dx = sKRK^-1
     */
    float I_x[3];
    mul_NTNT(1, 3, 3, (float*)I_x, (float*)I_y, (float*)cache->sKRKinv);

    /* d(pi^-1)/d(Dref) */
    float piinv_Dref[3];
    piinv_d(piinv_Dref, p_ref, cache->Dref[u_ref][v_ref]);

    *w = 2*cache->Ivar +
         square(I_x[0]*piinv_Dref[0] + I_x[1]*piinv_Dref[1] +
           I_x[2]*piinv_Dref[2]) * cache->Vref[u_ref][v_ref];
}
