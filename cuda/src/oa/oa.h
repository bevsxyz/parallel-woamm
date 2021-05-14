/// @file oa.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Enhanced Whale Optimization Algorithm
/// @version 1.0
/// @date 2021-05-09

#ifndef oa
#define oa

#include <vector>
#include "../../lib/mpgp32.cu"

using namespace std;
        
vector<float> run(float (*f)(float*), float l, float u);    
__global__ void woam(curandStateMtgp32 *devMTGPStates,float (*f)(float*),float l, float u, float*solution);

__device__ void getData(const int index,const float * __restrict__ myData,
const float * __restrict__ myCost,float * __restrict__ data,float * __restrict__ cost);

__device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest);

__device__ void updatePop(const int * __restrict__ random_particles,float * __restrict__ my_Data,
float * __restrict__ my_cost, curandStateMtgp32 *localState);

__device__ void msos(float (*f)(float*),float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState);

__device__ void woa(float (*f)(float*),float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState,
int current_iter,int * __restrict__ indexBest,float bound_low, float bound_high);


#endif //oa