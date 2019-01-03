#ifndef _LSDSLAM_H_
#define _LSDSLAM_H_

#include <stdbool.h>

#define WIDTH   240
#define HEIGHT  180

struct lsdslam_cache {
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

struct lsdslam_param {
    float mask_thresh;  /* use points which satisfy ||nabla I|| > mask_thresh */
    float huber_delta;  /* huber-norm */
};

struct lsdslam {
    struct lsdslam_param param;
    struct lsdslam_cache cache;
};

#endif
