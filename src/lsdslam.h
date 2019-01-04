#ifndef _LSDSLAM_H_
#define _LSDSLAM_H_

#include <stdbool.h>

#define WIDTH   240
#define HEIGHT  160

#ifdef BUILD_FOR_TEST
#define EXPORT
#else
#define EXPORT static
#endif

struct keyframe {
    float I[HEIGHT][WIDTH];
    float D[HEIGHT][WIDTH];
    float V[HEIGHT][WIDTH];
};

struct cache {
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
};

struct param {
    float initial_D;
    float initial_V;
    float mask_thresh;      /* use points which satisfy ||nabla I|| > mask_thresh */
    float huber_delta;      /* huber-norm */
    float K[3][3];          /* camera matrix */
};

struct tracker {
    struct param param;
    struct cache cache;
    struct keyframe keyframe;
    int frame;        /* number of processed frames */
    float eps;        /* epsilon for convergence test */
    int max_iter;     /* maximum number of iterations */
    float LMA_factor; /* factor v for Levenberg-Marquardt algorithm */
};

struct mapper {
    struct param param;
    struct cache cache;
};

#endif
