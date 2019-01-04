#include <stdio.h>
#include <stdlib.h>
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

/* matrix-matrix multiplication */
EXPORT void
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

EXPORT void
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

EXPORT float
iprod3d(float a[3], float b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

EXPORT void
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

EXPORT void
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

EXPORT void
mul3x3_twice(float y[3][3], float a[3][3], float b[3][3], float c[3][3])
{
    float t[3][3];
    mul3x3(t, b, c);
    mul3x3(y, a, t);
}

/* determinant of 3x3 matrix */
EXPORT float
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
EXPORT void
inv3x3(float y[3][3], float x[3][3])
{
    float det = det3x3(x);
    if (fabs(det) < 1e-12) {
        fprintf(stderr, "Singular matrix\n");
        exit(1);
    }
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

EXPORT void
affine3d(float y[3], float A[3][3], float b[3], float x[3])
{
    mulmv3d(y, A, x);
    for (int i = 0; i < 3; i++)
        y[i] += b[i];
}

/**** Rotation ****/

/* Rodrigues's rotation formula
 * Rotate x according to axis n and angle theta=|n|.
 * The result is stored to y */

EXPORT void
compute_nx(float y[3][3], float n[3])
{
    y[0][0] =     0; y[0][1] = -n[2]; y[0][2] =  n[1];
    y[1][0] =  n[2]; y[1][1] =     0; y[1][2] = -n[0];
    y[2][0] = -n[1]; y[2][1] =  n[0]; y[2][2] =     0;
}

EXPORT void
compute_R(float y[3][3], float n[3])
{
    y[0][0] = 1; y[0][1] = 0; y[0][2] = 0;
    y[1][0] = 0; y[1][1] = 1; y[1][2] = 0;
    y[2][0] = 0; y[2][1] = 0; y[2][2] = 1;

    float theta = sqrtf(iprod3d(n, n));

    if (fabs(theta) < 1e-30)
        return;

    float nx[3][3];
    float nx2[3][3];
    compute_nx(nx, n);
    mul3x3(nx2, nx, nx);

    float c1 = sinf(theta)/theta;
    float c2 = (1-cosf(theta))/(theta*theta);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            y[i][j] += c1*nx[i][j] + c2*nx2[i][j];
    }
}

// dR/dn
EXPORT void
compute_R_n(float y[3][3][3], float n[3])
{
    float theta = sqrtf(iprod3d(n, n));
    if (fabs(theta) < 1e-12) {
        memset(y, 0, sizeof(float)*3*3*3);
        // dR/dn1
        y[0][1][2] = -1;
        y[0][2][1] =  1;

        // dR/dn2
        y[1][0][2] =  1;
        y[1][2][0] = -1;

        // dR/dn2
        y[2][0][1] = -1;
        y[2][1][0] =  1;
        return;
    }


    float s = sinf(theta);
    float c = cosf(theta);
    float c1 = (theta * c - s) / (theta * theta * theta);
    float c2 = (theta * s + 2 *c - 2) / (theta * theta * theta * theta);
    float c3 = s / theta;
    float c4 = (1 - c) / (theta * theta);

    float nx[3][3];
    float nx2[3][3];
    compute_nx(nx, n);
    mul3x3(nx2, nx, nx);

    // dR/dn1
    y[0][0][0] = c1*n[0]*nx[0][0] + c2*n[0]*nx2[0][0];
    y[0][0][1] = c1*n[0]*nx[0][1] + c2*n[0]*nx2[0][1] + c4*n[1];
    y[0][0][2] = c1*n[0]*nx[0][2] + c2*n[0]*nx2[0][2] + c4*n[2];
    y[0][1][0] = c1*n[0]*nx[1][0] + c2*n[0]*nx2[1][0] + c4*n[1];
    y[0][1][1] = c1*n[0]*nx[1][1] + c2*n[0]*nx2[1][1] - 2*c4*n[0];
    y[0][1][2] = c1*n[0]*nx[1][2] + c2*n[0]*nx2[1][2] - c3;
    y[0][2][0] = c1*n[0]*nx[2][0] + c2*n[0]*nx2[2][0] + c4*n[2];
    y[0][2][1] = c1*n[0]*nx[2][1] + c2*n[0]*nx2[2][1] + c3;
    y[0][2][2] = c1*n[0]*nx[2][2] + c2*n[0]*nx2[2][2] - 2*c4*n[0];

    // dR/dn2
    y[1][0][0] = c1*n[1]*nx[0][0] + c2*n[1]*nx2[0][0] - 2*c4*n[1];
    y[1][0][1] = c1*n[1]*nx[0][1] + c2*n[1]*nx2[0][1] + c4*n[0];
    y[1][0][2] = c1*n[1]*nx[0][2] + c2*n[1]*nx2[0][2] + c3;
    y[1][1][0] = c1*n[1]*nx[1][0] + c2*n[1]*nx2[1][0] + c4*n[0];
    y[1][1][1] = c1*n[1]*nx[1][1] + c2*n[1]*nx2[1][1];
    y[1][1][2] = c1*n[1]*nx[1][2] + c2*n[1]*nx2[1][2] + c4*n[2];
    y[1][2][0] = c1*n[1]*nx[2][0] + c2*n[1]*nx2[2][0] - c3;
    y[1][2][1] = c1*n[1]*nx[2][1] + c2*n[1]*nx2[2][1] + c4*n[2];
    y[1][2][2] = c1*n[1]*nx[2][2] + c2*n[1]*nx2[2][2] - 2*c4*n[1];

    // dR/dn2
    y[2][0][0] = c1*n[2]*nx[0][0] + c2*n[2]*nx2[0][0] - 2*c4*n[2];
    y[2][0][1] = c1*n[2]*nx[0][1] + c2*n[2]*nx2[0][1] - c3;
    y[2][0][2] = c1*n[2]*nx[0][2] + c2*n[2]*nx2[0][2] + c4*n[0];
    y[2][1][0] = c1*n[2]*nx[1][0] + c2*n[2]*nx2[1][0] + c3;
    y[2][1][1] = c1*n[2]*nx[1][1] + c2*n[2]*nx2[1][1] - 2*c4*n[2];
    y[2][1][2] = c1*n[2]*nx[1][2] + c2*n[2]*nx2[1][2] + c4*n[1];
    y[2][2][0] = c1*n[2]*nx[2][0] + c2*n[2]*nx2[2][0] + c4*n[0];
    y[2][2][1] = c1*n[2]*nx[2][1] + c2*n[2]*nx2[2][1] + c4*n[1];
    y[2][2][2] = c1*n[2]*nx[2][2] + c2*n[2]*nx2[2][2];
}

EXPORT void
compute_identity(float *rho, float n[3], float t[3])
{
    *rho = 0.0;
    n[0] = 0.0;
    n[1] = 0.0;
    n[2] = 0.0;
    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 0.0;
}

/**** Projection to camera plane ****/
EXPORT void
pi(float y[3], float x[3])
{
    y[0] = x[0]/x[2];
    y[1] = x[1]/x[2];
    y[2] = 1/x[2];
}

EXPORT void
pip_x(float y[2][3], float x[3])
{
    y[0][0] = 1/x[2];
    y[0][1] = 0;
    y[0][2] = -x[0]/(x[2]*x[2]);
    y[1][0] = 0;
    y[1][1] = 1/x[2];
    y[1][2] = -x[1]/(x[2]*x[2]);
}

EXPORT void
piinv(float y[3], float x[2], float d)
{
    y[0] = x[0]/d;
    y[1] = x[1]/d;
    y[2] = 1/d;
}

EXPORT void
piinv_d(float y[3], float x[2], float d)
{
    y[0] = -x[0]/(d*d);
    y[1] = -x[1]/(d*d);
    y[2] = -1/(d*d);
}

EXPORT void
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

EXPORT void
gaussian_filter3x3(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        {1/9., 1/9., 1/9.},
        {1/9., 1/9., 1/9.},
        {1/9., 1/9., 1/9.},
    };
    filter3x3(y, k, x);
}

EXPORT void
gradu(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        { 1/4., 2/4., 1/4.},
        { 0., 0., 0.},
        {-1/4.,-2/4.,-1/4.},
    };
    filter3x3(y, k, x);
}

EXPORT void
gradv(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        {1/4., 0., -1/4.},
        {2/4., 0., -2/4.},
        {1/4., 0., -1/4.},
    };
    filter3x3(y, k, x);
}

EXPORT float
variance(float x[HEIGHT][WIDTH])
{
    float mean = 0.0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            mean += x[i][j];
        }
    }
    mean /= HEIGHT*WIDTH;
    float v = 0.0;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            v += square(x[i][j] - mean);
        }
    }
    return v / (HEIGHT*WIDTH);
}

/* Solve linear system Ax=b of degree n
 * NB: This function overwrite A and b
 */
EXPORT void
solve(float x[7], int dof, float A[7][7], float b[7])
{
    for (int k = 0; k < dof; k++) {
        /* select pivot */
        int pivot = k;
        float pmax = fabs(A[k][k]);
        for (int i = k+1; i < dof; i++) {
            float p = fabs(A[i][k]);
            if (p > pmax) {
                pivot = i;
                pmax = p;
            }
        }
        if (pmax < 1e-12) {
            fprintf(stderr, "Singular matrix\n");
            exit(1);
        }

        if (pivot != k) {
            /* swap row k and pivot */
            for (int j = 0; j < dof; j++) {
                float tmp = A[k][j];
                A[k][j] = A[pivot][j];
                A[pivot][j] = tmp;
            }
            float tmp = b[k];
            b[k] = b[pivot];
            b[pivot] = tmp;
        }

        for (int i = k+1; i < dof; i++) {
            float c = A[i][k]/A[k][k];
            for (int j = k+1; j < dof; j++)
                A[i][j] -= c*A[k][j];
            A[i][k] = 0;
            b[i] -= c*b[k];
        }
    }
    for (int i = dof-1; i >= 0; i--) {
        float tmp = b[i];
        for (int j = dof-1; j > i; j--)
            tmp -= x[j] * A[i][j];
        x[i] = tmp / A[i][i];
    }
}

EXPORT void
create_mask(bool mask[HEIGHT][WIDTH], float I[HEIGHT][WIDTH], float thresh)
{
    float I_[HEIGHT][WIDTH];
    float gu[HEIGHT][WIDTH];
    float gv[HEIGHT][WIDTH];

    gaussian_filter3x3(I_, I);
    gradu(gu, I_);
    gradv(gv, I_);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++)
            mask[i][j] = sqrtf(square(gu[i][j]) + square(gv[i][j])) > thresh;
    }
}

EXPORT float
huber(float delta, float r)
{
    if (r > delta)
        return r - delta/2;
    else if (r < -delta)
        return -r - delta/2;
    else
        return r*r/(2*delta);
}

EXPORT float
huber_r(float delta, float r)
{
    if (r > delta)
        return 1;
    else if (r < -delta)
        return -1;
    else
        return r/delta;
}

EXPORT void
precompute_cache(
        struct param *param,
        struct cache *cache,
        float Iref[HEIGHT][WIDTH],
        float Dref[HEIGHT][WIDTH],
        float Vref[HEIGHT][WIDTH],
        float I[HEIGHT][WIDTH],
        float rho,
        float n[3],
        float t[3]
        )
{
    create_mask(cache->mask, Iref, param->mask_thresh);

    memcpy(cache->Iref, Iref, sizeof(cache->Iref));
    memcpy(cache->Dref, Dref, sizeof(cache->Dref));
    memcpy(cache->Vref, Vref, sizeof(cache->Vref));

    memcpy(cache->I, I, sizeof(cache->I));
    gradu(cache->I_u, I);
    gradv(cache->I_v, I);
    cache->Ivar = variance(I);

    float Kinv[3][3];
    inv3x3(Kinv, param->K);

    float sR[3][3];
    float s = expf(rho);
    compute_R(sR, n);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            sR[i][j] *= s;

    mul3x3_twice(cache->sKRKinv, param->K, sR, Kinv);
    mulmv3d(cache->Kt, param->K, t);

    float sR_n[3][3][3];
    compute_R_n(sR_n, n);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                sR_n[i][j][k] *= s;
    mul3x3_twice(cache->sKR_nKinv[0], param->K, sR_n[0], Kinv);
    mul3x3_twice(cache->sKR_nKinv[1], param->K, sR_n[1], Kinv);
    mul3x3_twice(cache->sKR_nKinv[2], param->K, sR_n[2], Kinv);
}

// Compute photometric residual, derivative wrt xi and weight.
// *res will be NaN for out-bound error.
EXPORT int
photometric_residual(
        struct cache *cache,
        int dof, float *rp, float *wp, float J[7],
        int u_ref, int v_ref)
{
    float p_ref[2] = {u_ref, v_ref};
    float x[3];
    float y[3];
    float q[3];

    piinv(x, p_ref, cache->Dref[u_ref][v_ref]);
    affine3d(y, cache->sKRKinv, cache->Kt, x);
    pi(q, y);

    int u = (int)q[0];
    int v = (int)q[1];

    if (isnan(q[0]) || isnan(q[1]) || u < 0 || u >= HEIGHT || v < 0 || v >= WIDTH) {
        *rp = NAN;
        return -1;
    }

    //printf("%d, %d, %d, %d\n", u_ref, v_ref, u, v);

    *rp = cache->Iref[u_ref][v_ref] - cache->I[u][v];

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
    float tau_xi_T[7][3];

    // d(tau)/d(t)(x) = I
    tau_xi_T[0][0] = 1;
    tau_xi_T[0][1] = 0;
    tau_xi_T[0][2] = 0;
    tau_xi_T[1][0] = 0;
    tau_xi_T[1][1] = 1;
    tau_xi_T[1][2] = 0;
    tau_xi_T[2][0] = 0;
    tau_xi_T[2][1] = 0;
    tau_xi_T[2][2] = 1;
    mulmv3d(tau_xi_T[3], cache->sKR_nKinv[0], x);  // d(tau)/d(n_1)(x) = sKR_n_1K^-1x
    mulmv3d(tau_xi_T[4], cache->sKR_nKinv[1], x);  // d(tau)/d(n_2)(x) = sKR_n_2K^-1x
    mulmv3d(tau_xi_T[5], cache->sKR_nKinv[2], x);  // d(tau)/d(n_3)(x) = sKR_n_3K^-1x
    if (dof == 7)
        mulmv3d(tau_xi_T[6], cache->sKRKinv, x);       // d(tau)/d(rho)(x) = sKRK^-1x 

    // J = -dI/d(xi)
    mul_NTT(1, 3, dof, (float*)J, (float*)I_y, (float*)tau_xi_T);
    for (int i = 0; i < dof; i++)
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
    float I_Dref = I_x[0]*piinv_Dref[0] + I_x[1]*piinv_Dref[1] + I_x[2]*piinv_Dref[2];

    *wp = 1/(2*cache->Ivar + square(I_Dref) * cache->Vref[u_ref][v_ref]);
    return 0;
}

/* Compute E_p, g = nabla E_p, H = nabla^2 E_p */
EXPORT void
photometric_residual_over_frame(
        struct param *param,
        struct cache *cache,
        int dof, float *E, float g[7], float H[7][7]
        )
{
    int N = 0;

    *E = 0;
    memset(g, 0, sizeof(float)*7);
    memset(H, 0, sizeof(float)*49);

    float rp;
    float wp;
    float J[7];
    for (int u = 0; u < HEIGHT; u++) {
        for (int v = 0; v < WIDTH; v++) {
            if (!cache->mask[u][v])
                continue;

            if (photometric_residual(cache, dof, &rp, &wp, J, u, v) < 0)
                continue;

            *E += wp * huber(param->huber_delta, rp);

            for (int i = 0; i < dof; i++)
                g[i] += wp * huber_r(param->huber_delta, rp) * J[i];

            if (fabs(rp) < param->huber_delta) {
                for (int i = 0; i < dof; i++) {
                    for (int j = 0; j < dof; j++)
                        H[i][j] += wp*J[i]*J[j];
                }
            }

            N++;
        }
    }

    *E /= N;
    for (int i = 0; i < dof; i++)
        g[i] /= N;
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++)
            H[i][j] /= N * param->huber_delta;
    }
}

EXPORT void
copy_image(float y[HEIGHT][WIDTH], unsigned char x[HEIGHT][WIDTH])
{
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++)
            y[i][j] = x[i][j];
    }
}

/**** Tracking ****/
struct tracker *
allocate_tracker(void)
{
    struct tracker *obj = malloc(sizeof(struct tracker));
    if (!obj) {
        perror("Failed to allocate memory");
        exit(1);
    }
    memset(obj, 0, sizeof(struct tracker));
    return obj;
}

void
release_tracker(struct tracker *obj)
{
    free(obj);
}

void
tracker_init(
        struct tracker *tracker,
        float initial_D,
        float initial_V,
        float mask_thresh,
        float huber_delta,
        float K[3][3],
        float eps,
        int max_iter,
        float levmar_factor
        )
{
    tracker->frame = 0;
    tracker->eps = eps;
    tracker->levmar_factor = levmar_factor;
    tracker->max_iter = max_iter;
    tracker->param.initial_D = initial_D;
    tracker->param.initial_V = initial_V;
    tracker->param.mask_thresh = mask_thresh;
    tracker->param.huber_delta = huber_delta;
    memcpy(tracker->param.K, K, sizeof(tracker->param.K));
}

static void
set_initial_frame(struct tracker *tracker, unsigned char image[HEIGHT][WIDTH])
{
    copy_image(tracker->keyframe.I, image);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            tracker->keyframe.D[i][j] = tracker->param.initial_D;
            tracker->keyframe.V[i][j] = tracker->param.initial_V;
        }
    }
}

void
tracker_estimate(
        struct tracker *tracker,
        unsigned char image[HEIGHT][WIDTH],
        float n[3], float t[3]
        )
{
    struct timeval start, end, elapsed;
    gettimeofday(&start, NULL);

    float rho;
    compute_identity(&rho, n, t);
    if (tracker->frame == 0) {
        set_initial_frame(tracker, image);
        tracker->frame++;
        return;
    }

    float I[HEIGHT][WIDTH];
    copy_image(I, image);
    float xi[7] = {
        n[0], n[1], n[2], t[0], t[1], t[2], 0.0 /* rho */
    };
    float *n_ = xi;
    float *t_ = xi + 3;
    float delta_xi[7];
    float g[7];
    float H[7][7];

    float prevE = 1e10;

    /* damping factor (Levenberg-Marquardt algorithm) */
    float lambda = 0.0;

    for (int i = 0; i < tracker->max_iter; i++) {
        precompute_cache(
                &tracker->param, &tracker->cache,
                tracker->keyframe.I, tracker->keyframe.D, tracker->keyframe.V,
                I, 0.0, n_, t_);

        float E = 0;

        /* E_p component */
        photometric_residual_over_frame(
                &tracker->param, &tracker->cache,
                6, &E, g, H);

        /* Add lambda*I to avoid being singular matrix */
        for (int i = 0; i < 6; i++)
            H[i][i] *= 1.2;

        solve(delta_xi, 6, H, g);

        for (int i = 0; i < 6; i++)
            xi[i] -= delta_xi[i];

        if (fabs((E-prevE)/prevE) < tracker->eps) {
            printf("iteration=%d\n", i+1);
            break;
        }
        prevE = E;
    }
    printf("E=%f\n", prevE);

    tracker->frame++;

    gettimeofday(&end, NULL);
    timersub(&end, &start, &elapsed);
    printf("%fms\n", elapsed.tv_usec/1.0e3);

    t[0] = xi[0];
    t[1] = xi[1];
    t[2] = xi[2];
    n[0] = xi[3];
    n[1] = xi[4];
    n[2] = xi[5];
}
