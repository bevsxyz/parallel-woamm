/// @file func.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Math functions
/// @version 1.0
/// @date 2021-05-08

#ifndef func_h
#define func_h

namespace func {
    // Variable dimensions unimodal functions

    __device__ float sphere(const float * __restrict__ x);
    __device__ float rosenbrock(const float * __restrict__ x);

    // Variable dimensions multimodal functions

    __device__ float rastrigin(const float * __restrict__ x);
    __device__ float griewangk(const float * __restrict__ x);
}

#endif

// =========================
// End of func.h
// =========================