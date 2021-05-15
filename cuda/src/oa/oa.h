/// @file oa.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Enhanced Whale Optimization Algorithm
/// @version 1.0
/// @date 2021-05-09

#ifndef oa
#define oa

#include <vector>
#include <curand.h>
#include <curand_kernel.h>
/// include MTGP host helper functions
#include <curand_mtgp32_host.h>
/// include MTGP pre-computed parameter sets
#include <curand_mtgp32dc_p_11213.h>
#include "../func/func.h"

using namespace std;
        
vector<float> run(int f, float l, float u);    
__global__ void woam(curandStateMtgp32 *devMTGPStates,int f,float l, float u, float*solution);

__device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest);

__device__ void updatePop(int f,const int * __restrict__ random_particles,float * __restrict__ my_Data,
float * __restrict__ my_cost, curandStateMtgp32 *localState);

__device__ void msos(int f,float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState);

__device__ void woa(int f,float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState,
int current_iter,int * __restrict__ indexBest,float bound_low, float bound_high);


#endif //oa