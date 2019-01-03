#ifndef _LSDSLAM_H_
#define _LSDSLAM_H_

#include <stdbool.h>

#define WIDTH   240
#define HEIGHT  180

struct compute_cache {
    /* keyframe */
    float Iref[HEIGHT][WIDTH];
    float Dref[HEIGHT][WIDTH];
    float Vref[HEIGHT][WIDTH];
    bool  Mref[HEIGHT][WIDTH];

    /* current frame */
    float I[HEIGHT][WIDTH];
    float I_x[HEIGHT][WIDTH];
    float I_y[HEIGHT][WIDTH];
};

#endif
