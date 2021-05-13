#include "oa.h"

/// Initialise the host and device parameters
/// Setup MTGP prng states
/// @param f Function to be evaluvated
/// @param l lower bound of the function
/// @param u upper bound of the function
OA::OA(__device__ float (*f)(const float __restrict__ &), float l, float u){
    function = f;
    bound_low = l;
    bound_high = u;
    cudaMalloc((void**) &device_solution, (sizeof(float)));

    /// Allocate space for prng states on device
    cudaMalloc((void **)&devMTGPStates, dimension * 
    sizeof(curandStateMtgp32));
     
    /// Allocate space for MTGP kernel parameters
    cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));

    /// Reformat from predefined parameter sets to kernel format,
    /// and copy kernel parameters to device memory
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
    
    /// Initialize one state per thread block
    curandMakeMTGP32KernelState(devMTGPStates, 
                mtgp32dc_params_fast_11213, devKernelParams, psize, time(NULL));
}

__global__ void OA::woam(){
    curandStateMtgp32 localState = devMTGPStates[threadIdx.x]
    float myData[dimension],cost,costBest;
    int indexBest=threadIdx.x;

    for (int j = 0; j < dimension; j++){
        /// generate data
        myData[j] = (float)(bound_low + (curand_uniform(&localState)* (bound_high - bound_low));
    }
    cost = function(myData);
    costBest = cost;

    getBest(&indexBest,&costBest);

    for (int k = 0; k < max_iter; k++){
        // mMSOS Component
        msos(&myData,&cost,&localState,&indexBest,&costBest);

        // WOA Component
        woa(&myData,&cost,k,&localState,&indexBest,&costBest);
    }

}

/// Finds the best cost in the population
/// Uses a butterfly reduction
/// @param indexBest Index of the individual with minimum cost, will be updated
/// @param costBest Cost of the individual with minimum cost, will be updated
__device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest){
    float costTemp;
    int indexTemp;
    for (int i=16; i>=1; i/=2){
        costTemp = __shfl_xor_sync(0xffffffff, costBest, i, 32);
        indexTemp = __shfl_xor_sync(0xffffffff, indexBest, i,32);
        if(costTemp<costBest){
            costBest=costTemp;
            indexBest = indexTemp;
        }
    }
}

/// Get data from individual "index"
/// @param index Index of the individual to copy data from
/// @param myData Pointer for my data
/// @param myCost Pointer for my cost
/// @param data Pointer to the float array to which we will copy the other individual's data
/// @param cost Pointer to the float to which we will copy the other individual's cost
__device__ void OA::getData(const int index,const float * __restrict__ myData,const float * __restrict__ myCost,
    float * __restrict__ data,float * __restrict__ cost){
    *cost = __shfl_sync(0xffffffff, myCost,index);
    for (int i = 0; i < dimension; i++){
        data[i] = __shfl_sync(0xffffffff, myData[i],index);
    }
}

/// Component Optimization Algorithm: Modified Mutualism Phase of SOS
/// @param myData Vector array of data of individual
/// @param cost Cost of myData for the given function
/// @param localState MTGP32 PRG state
/// @param indexBest Index of the individual with minimum cost
/// @param costBest Cost of the individual with minimum cost
__device__ void OA::msos(float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState,
    int * __restrict__ indexBest,float * __restrict__ costBest){
    int random_particles,pick,m,n;
    float cost_rp[2],rp0Data[dimension],rp1Data[dimension];
    for (int i = 0; i < psize; i++){
        random_particles[0] = int(curand_uniform(localState) * (psize-1))-1;
        if(random_particles[0] >= i)
            random_particles[0]++;
        random_particles[1] = int(curand_uniform(localState) * (psize-2))-1;
        if(random_particles[1] >= i)
            random_particles[1]++;
        if(random_particles[1] >= random_particles[0])
            random_particles[1]++;
        
        getData(random_particles[0],my_Data,my_cost,rp0Data,&cost_rp[0]);
        getData(random_particles[1],my_Data,my_cost,rp1Data,&cost_rp[1]);

        pick = cost_rp[0] < cost_rp[1];
    }
}


__device__ void OA::woa(float * __restrict__ cost,int k,curandStateMtgp32 *localState){

}

vector<float> OA::run(){
    vector<float> global_best_solution;

    woam<<1,psize>>(bound_low,bound_high,time(NULL));

    global_best_solution.pushback(*best_solution);
    return global_best_solution;
}