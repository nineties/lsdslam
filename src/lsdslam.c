#include <stdio.h>
#include <math.h>

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
