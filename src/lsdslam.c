#include <stdio.h>
#include <string.h>
#include <math.h>
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

/* Ax + b = K(sRx + t) */
void
precompute_KT(float A[3][3], float b[3],
        float K[3][3], float rho, float n[3], float theta, float t[3])
{
    float _A[3][3];
    float _b[3];
    precompute_T(_A, _b, rho, n, theta, t);
    mul3x3(A, K, _A);
    mulmv3d(b, K, _b);
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
sobelx(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        {1., 0., -1.},
        {2., 0., -2.},
        {1., 0., -1.},
    };
    filter3x3(y, k, x);
}

void
sobely(float y[HEIGHT][WIDTH], float x[HEIGHT][WIDTH])
{
    float k[3][3] = {
        { 1., 2., 1.},
        { 0., 0., 0.},
        {-1.,-2.,-1.},
    };
    filter3x3(y, k, x);
}

void
create_mask(bool mask[HEIGHT][WIDTH], float I[HEIGHT][WIDTH], float thresh)
{
    float gx[HEIGHT][WIDTH];
    float gy[HEIGHT][WIDTH];
    sobelx(gx, I);
    sobely(gy, I);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++)
            mask[i][j] = sqrtf(square(gx[i][j]) + square(gy[i][j])) > thresh;
    }
}

void
precompute_cache(
        struct lsdslam *slam,
        float Iref[HEIGHT][WIDTH],
        float Dref[HEIGHT][WIDTH],
        float Vref[HEIGHT][WIDTH],
        float I[HEIGHT][WIDTH]
        )
{
    memcpy(slam->cache.Iref, Iref, sizeof(slam->cache.Iref));
    memcpy(slam->cache.Dref, Dref, sizeof(slam->cache.Dref));
    memcpy(slam->cache.Vref, Vref, sizeof(slam->cache.Vref));
    create_mask(slam->cache.mask, Iref, slam->param.mask_thresh);

#if 0
    {
        float k[3][3] = {
            {0., 0., 0.};
            {0., 0., 0.};
            {0., 0., 0.};

        slam->cache.I_u
#endif
}

