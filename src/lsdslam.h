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

struct cache {
    /* keyframe */
    int Nref;   /* number of points in keyframe */
    int (*pref)[2];
    float *Iref;
    float *Dref;
    float *Vref;
    float (*piinv)[3];      /* pi^-1 */
    float (*piinv_Dref)[3]; /* d(pi^-1)/d(Dref) */

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

enum optimize_algo {
    OPTIMIZE_LMA,
    OPTIMIZE_BFGS,
};

struct tracker {
    struct param param;
    struct cache cache;
    int frame;          /* number of processed frames */
    float eps;          /* epsilon for convergence test */
    int max_iter;       /* maximum number of iterations */
    float LMA_lambda0;  /* initial lambda for Levenberg-Marquardt algorithm */
    float LMA_scale;    /* scaling factor v for Levenberg-Marquardt algorithm */
    float min_pixel_usage;
    float step_size_min;
    enum optimize_algo optimize_algo;
};

struct mapper {
    struct param param;
    struct cache cache;
};

#endif
