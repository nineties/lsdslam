#include <stdio.h>
#include <math.h>

/**** Rotation ****/

/* Rodrigues's rotation formula
 * Rotate x according to axis n and angle theta.
 * The result is stored to y */
void
R(float y[3][3], float n[3], float theta)
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
R_theta(float y[3][3], float n[3], float theta)
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
R_n(float y[3][3][3], float n[3], float theta)
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
