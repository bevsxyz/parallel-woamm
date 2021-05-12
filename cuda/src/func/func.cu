#include "func.h"

#define fpsize 30
#define PI 3.14159265358979323846

/// Function 1, Implementation of Sphere function
/// @param x Array vector of float
__device__ float func::sphere(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize; i++){
        result += x[i] * x[i];
    }
    return result;
}

/// Function 2, Implementation of Rosenbrock's function
/// @param x Array vector of float
__device__ float func::rosenbrock(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize - 1; i++){
        double a = x[i] * x[i] - x[i+1];
        double b = 1.0 - x[i];
        result += 100 * a * a + b * b;
    }
    return result;
}

/// Function 3, Implementation of Rastrigin's function
/// @param x Array vector of float
__device__ float func::rastrigin(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize ; i++){
        result += x[i] * x[i] - 10.0 * __cosf((2 * PI * x[i]));
    }
    return 10 * fpsize + result;
}

/// Function 4, Implementation of Griewangk function
__device__ float func::griewangk(const float * __restrict__ x){
    float sum = 0;
    float product = 1;
    for (int i = 0; i < fpsize ; i++){
        sum += x[i] * x[i] / 4000;
        product *= __cosf(x[i]*rsqrtf(i+1));
    }
    return 1.0 + sum - product;
}