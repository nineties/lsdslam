#ifndef _LSDSLAM_H_
#define _LSDSLAM_H_

#include <stdbool.h>

#define WIDTH   240
#define HEIGHT  180

struct compute_cache {
    bool  mask[HEIGHT][WIDTH];

    /* keyframe */
    float Iref[HEIGHT][WIDTH];
    float Dref[HEIGHT][WIDTH];
    float Vref[HEIGHT][WIDTH];

    /* current frame */
    float I[HEIGHT][WIDTH];
    float I_u[HEIGHT][WIDTH];
    float I_v[HEIGHT][WIDTH];
    float Ivar;

    /* coefficients for d(rp)/d(xi) */
    float Kt[3];
    float sKRKinv[3][3];
    float sKR_nKinv[3][3][3];
    float sKR_thetaKinv[3][3];
};

struct param {
    float mask_thresh;
};

struct lsdslam {
    struct param param;
    struct compute_cache cache;
};

#endif
