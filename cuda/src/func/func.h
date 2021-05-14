/// @file func.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Math functions
/// @version 1.0
/// @date 2021-05-08

#ifndef func_h
#define func_h

namespace func {
    // Variable dimensions unimodal functions

    float sphere(float * x);
    float rosenbrock(float * x);

    // Variable dimensions multimodal functions

    float rastrigin(float * x);
    float griewangk(float * x);
}

#endif

// =========================
// End of func.h
// =========================