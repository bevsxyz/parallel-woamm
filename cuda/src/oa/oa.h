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
#include <thrust/device_vector.h>
/// include MTGP host helper functions
#include <curand_mtgp32_host.h>
/// include MTGP pre-computed parameter sets
#include <curand_mtgp32dc_p_11213.h>

using namespace std;

class OA{
    public:
        OA(__device__ float (*f)(const float __restrict__ &), float l, float u);
        vector<float> run();        

    private:
        /// Variables

        int max_iter = 30;       /// Max iteration
        float b = 0.8;
        int psize = 32;          /// Size of population
        int dimension = 30;      /// Dimension
        float host_solution;     /// Host pointer for best cost
        float * device_solution; /// Device pointer for best cost

        float bound_low; /// Objective function solution lower bound
        float bound_high; /// Objective function solution higher bound

        curandStateMtgp32 *devMTGPStates;      /// State array for MTGP32 generator
        mtgp32_kernel_params *devKernelParams; /// Parameters for initialising PRG

        
        

        /// Functions

        __device__ float (*function) (const float __restrict__ &);
        __device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest);
        __device__ void msos(float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState,
        int * __restrict__ indexBest,float * __restrict__ costBest);
        __global__ void woam();

};


#endif //oa