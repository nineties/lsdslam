#include <stdio.h>
#include <math.h>


/* Rodrigues's rotation formula
 * Rotate x according to axis n and angle theta.
 * The result is stored to y */
void
R(float y[3], float n[3], float theta, float x[3])
{
    float s = sinf(theta);
    float c = cosf(theta);

    y[0] = (c + n[0]*n[0]*(1-c))*x[0] +
           (n[0]*n[1]*(1-c) - n[2]*s)*x[1] +
           (n[2]*n[0]*(1-c) + n[1]*s)*x[2];
    y[1] = (n[0]*n[1]*(1-c) + n[2]*s)*x[0] +
           (c + n[1]*n[1]*(1-c))*x[1] +
           (n[1]*n[2]*(1-c) - n[0]*s)*x[2];
    y[2] = (n[2]*n[0]*(1-c) - n[1]*s)*x[0] +
           (n[1]*n[2]*(1-c) + n[0]*s)*x[1] +
           (c + n[2]*n[2]*(1-c))*x[2];
}

/* d(R)/d(theta) */
void
R_theta(float y[3], float n[3], float theta, float x[3])
{
    float s = sinf(theta);
    float c = cosf(theta);

    y[0] = (-s + n[0]*n[0]*s)*x[0] +
           (n[0]*n[1]*s - n[2]*c)*x[1] +
           (n[2]*n[0]*s + n[1]*c)*x[2];
    y[1] = (n[0]*n[1]*s + n[2]*c)*x[0] +
           (-s + n[1]*n[1]*s)*x[1] +
           (n[1]*n[2]*s - n[0]*c)*x[2];
    y[2] = (n[2]*n[0]*s - n[1]*c)*x[0] +
           (n[1]*n[2]*s + n[0]*c)*x[1] +
           (-s + n[2]*n[2]*s)*x[2];
}

/* d(R)/d(n) */
void
R_n(float y[3][3], float n[3], float theta, float x[3])
{
    float s = sinf(theta);
    float c = cosf(theta);
    /* sin(theta)*x_x */
    y[0][0] =       0; y[0][1] = -s*x[2]; y[0][2] = s*x[1];
    y[1][0] =  s*x[2]; y[1][1] =       0; y[1][2] = -s*x[0];
    y[2][0] = -s*x[1]; y[2][1] =  s*x[0]; y[2][2] = 0;

    /* (1-cos(theta))xn^T */
    y[0][0] += (1-c)*x[0]*n[0]; y[0][1] += (1-c)*x[0]*n[1]; y[0][2] += (1-c)*x[0]*n[2];
    y[1][0] += (1-c)*x[1]*n[0]; y[1][1] += (1-c)*x[1]*n[1]; y[1][2] += (1-c)*x[1]*n[2];
    y[2][0] += (1-c)*x[2]*n[0]; y[2][1] += (1-c)*x[2]*n[1]; y[2][2] += (1-c)*x[2]*n[2];

    /* (1-cos(theta))(x^Tn)I */
    float t = (1-c)*(x[0]*n[0] + x[1]*n[1] + x[2]*n[2]);
    y[0][0] += t;
    y[1][1] += t;
    y[2][2] += t;
}
