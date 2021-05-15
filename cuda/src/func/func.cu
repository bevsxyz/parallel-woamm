#include "func.h"

#define fpsize 30
#define PI 3.14159265358979323846

__device__ void func(int f, const float * __restrict__ mydata,float * __restrict__ cost){
    if(f==1){
        *cost = sphere(mydata);
    }
    else if(f==2){
        *cost = rosenbrock(mydata);
    }
    else if(f==3){
        *cost = rastrigin(mydata);
    }     
    else if(f==4){
        *cost = griewangk(mydata);
    }
    else if(f==5){
        *cost = f8(mydata);
    }
    else if(f==6){
        *cost = brown(mydata);
    }
}

/// Function 1, Implementation of Sphere function
/// @param x Array vector of float
__device__ float sphere(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize; i++){
        result += x[i] * x[i];
    }
    return result;
}

/// Function 2, Implementation of Rosenbrock's function
/// @param x Array vector of float
__device__ float rosenbrock(const float * __restrict__ x){
    float result = 0, a,b;
    for (int i = 0; i < fpsize - 1; i++){
        a = x[i] * x[i] - x[i+1];
        b = 1.0 - x[i];
        result += 100 * a * a + b * b;
    }
    return result;
}

/// Function 3, Implementation of Rastrigin's function
/// @param x Array vector of float
__device__ float rastrigin(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize ; i++){
        result += x[i] * x[i] - 10.0 * __cosf(2 * PI * x[i]);
    }
    return 10 * fpsize + result;
}

/// Function 4, Implementation of Griewangk function
__device__ float griewangk(const float * __restrict__ x){
    float sum = 0;
    float product = 1;
    for (int i = 0; i < fpsize ; i++){
        sum += x[i] * x[i] / 4000;
        product *= __cosf(x[i]*rsqrtf(i+1));
    }
    return 1.0 + sum - product;
}

__device__ float f8(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize ; i++){
        result -= x[i] * __sinf(sqrtf(fabsf(x[i])));
    }
    return result;
}

__device__ float brown(const float * __restrict__ x){
    float result = 0;
    float xp12;
    for (int i = 0; i < fpsize-1 ; i++){
        xp12 = x[i+1]*x[i+1];
        result += __powf(x[i]*x[i],(xp12+1)) + __powf(xp12,xp12+1);
    }
    return result;
}