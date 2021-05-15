#include "oa.h"

#define max_iter 30
#define psize 32
#define dimension 30
#define PI 3.14159265358979323846
#define b 0.8

/// @param f Function to be evaluvated
/// @param l lower bound of the function
/// @param u upper bound of the function
vector<float> run(int f, float l, float u){

    int blocks = 6;
    int threads = 32;
    vector<float> global_best_solution;
    curandStateMtgp32 *devMTGPStates;      /// State array for MTGP32 generator
    mtgp32_kernel_params *devKernelParams; /// Parameters for initialising PRG

    float host_solution[blocks],*device_solution;
    cudaMalloc((void**)&device_solution, blocks* sizeof(float));

    /// Allocate space for prng states on device
    cudaMalloc((void **)&devMTGPStates, blocks*threads * sizeof(curandStateMtgp32));
    
    /// Allocate space for MTGP kernel parameters
    cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));

    /// Reformat from predefined parameter sets to kernel format,
    /// and copy kernel parameters to device memory
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
    
    /// Initialize one state per thread block
    curandMakeMTGP32KernelState(devMTGPStates, 
                mtgp32dc_params_fast_11213, devKernelParams, blocks*threads, time(NULL));

    woam<<<blocks,threads>>>(devMTGPStates,f,l,u, device_solution);

    cudaMemcpy(host_solution, device_solution, blocks* (sizeof(float)), cudaMemcpyDeviceToHost);
    
    float best = host_solution[0];
    for(int i = 1; i < blocks; i++){
        if(host_solution[i]<best)
            best = host_solution[i];
    }
    global_best_solution.push_back(best);
    cudaFree(device_solution);
    cudaFree(devMTGPStates);
    cudaFree(devKernelParams);
    
    return global_best_solution;
}

/// The main function for WOAM
/// @param f Function to be evaluvated
/// @param l lower bound of the function
/// @param u upper bound of the function
__global__ void woam(curandStateMtgp32 *devMTGPStates,int f,float l, float u, float*solution){
    int myID = threadIdx.x;
    int bID  = blockIdx.x;
    curandStateMtgp32 localState = devMTGPStates[(bID*32)+myID];
    float myData[dimension],cost,costBest;
    int indexBest=myID;

    for (int j = 0; j < dimension; j++){
        /// generate data
        myData[j] = (float)(l + (curand_uniform(&localState)* (u - l)));
    }
    func(f,&myData[0],&cost);
    costBest = cost;

    getBest(&indexBest,&costBest);

    for (int k = 0; k < max_iter; k++){
        // mMSOS Component
        msos(f,&myData[0],&cost,&localState);

        costBest = cost;
        indexBest=myID;
        getBest(&indexBest,&costBest);

        // WOA Component
        woa(f,&myData[0],&cost,&localState,k,&indexBest, l,u);

        costBest = cost;
        indexBest=myID;
        getBest(&indexBest,&costBest);
    }
    if(myID==0)
        solution[bID] = costBest;
}

/// Finds the best cost in the population
/// Uses a butterfly reduction
/// @param indexBest Index of the individual with minimum cost, will be updated
/// @param costBest Cost of the individual with minimum cost, will be updated
__device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest){\
    int ID = *indexBest;
    float Best = *costBest;
    float costTemp;
    int indexTemp;
    for (int i=16; i>=1; i/=2){
        costTemp = __shfl_xor_sync(0xffffffff, Best, i, 32);
        indexTemp = __shfl_xor_sync(0xffffffff, ID, i,32);
        if(costTemp<Best){
            Best = costTemp;
            ID = indexTemp;
        }
    }
    *costBest=Best;
    *indexBest = ID;
}

/// Get data from individual "index"
/// @param index Index of the individual to copy data from
/// @param myData Pointer for my data
/// @param myCost Pointer for my cost
/// @param data Pointer to the float array to which we will copy the other individual's data
/// @param cost Pointer to the float to which we will copy the other individual's cost
__device__ void getData(const int index,const float * __restrict__ myData,const float * __restrict__ myCost,
    float * __restrict__ data,float * __restrict__ cost){
    *cost = __shfl_sync(0xffffffff, *myCost,index);
    for (int i = 0; i < dimension; i++){
        data[i] = __shfl_sync(0xffffffff, myData[i],index);
    }
}

/// Update the population for msos
/// @param random_particles Array of the two random individuals picked
/// @param myData Pointer for my data
/// @param myCost Pointer for my cost
/// @param localState MTGP32 PRG state
__device__ void updatePop(int f,const int * __restrict__ random_particles,float * __restrict__ my_Data,
    float * __restrict__ my_cost, curandStateMtgp32 *localState){
    
    float cost_rp[2],data_rp[2*30];

    getData(random_particles[0],my_Data,my_cost,data_rp,&cost_rp[0]);
    getData(random_particles[1],my_Data,my_cost,&data_rp[dimension],&cost_rp[1]);

    /// Calculate Fitness
    int highFitness, lowFitness, hIndex , lIndex;

    highFitness = cost_rp[0] < cost_rp[1];
    lowFitness = !highFitness;

    /// Also the corresponding starting indices of the high and low fitness individuals
    hIndex = highFitness * dimension;
    lIndex = lowFitness * dimension;

    float my_Data_kp1[dimension];
    float my_cost_kp1;
    
    int bf1,bf2,x,y,z;
    float mv;

    bf1 = 1 + curand_uniform(localState) * 2;
    bf2 = 1 + curand_uniform(localState) * 2;

    /// Calculate the k+1 population values
    for (x = 0,y=lIndex,z=hIndex; x < dimension; x++,y++,z++){
        mv = __fdividef((my_Data[x] + data_rp[z]),2);
        my_Data_kp1[x] = my_Data[x] + (curand_uniform(localState) * (data_rp[y] - mv*bf1));
        data_rp[x] = data_rp[z]  + (curand_uniform(localState) * (data_rp[y] - mv*bf2));
    }

    /// Calculate the costs
    func(f,my_Data_kp1,&my_cost_kp1);
    func(f,data_rp,&cost_rp[0]);

    /// If the new cost is the minima update my individual data
    if(my_cost_kp1 < *my_cost){
        *my_cost = my_cost_kp1;
        for(int i = 0; i < dimension; i++)
            my_Data[i] = my_Data_kp1[i];
    }

    /// Need to implement the update of random indivdual
    int myID = threadIdx.x;
    int index=myID,indexTemp, rindex=random_particles[highFitness];
    float costTemp;
    for(int i = 0; i < 32; i++){
        costTemp = __shfl_sync(0xffffffff, cost_rp[0],i);
        indexTemp = __shfl_sync(0xffffffff, rindex,i);
        if(indexTemp == myID){
            if(*my_cost>costTemp){
                index = i;
                *my_cost = costTemp;
            }
        }
    }

    for (int i = 0; i < dimension; i++){
        my_Data[i] = __shfl_sync(0xffffffff, data_rp[i],index);
    }

}

/// Component Optimization Algorithm: Modified Mutualism Phase of SOS
/// @param myData Vector array of data of individual
/// @param cost Cost of myData for the given function
/// @param localState MTGP32 PRG state
__device__ void msos(int f,float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState){
    int random_particles[2];
    int my_index = threadIdx.x;
    random_particles[0] = int(curand_uniform(localState) * (psize-1))-1;
    if(random_particles[0] >= my_index)
        random_particles[0]++;
    random_particles[1] = int(curand_uniform(localState) * (psize-2))-1;
    if(random_particles[1] >= my_index)
        random_particles[1]++;
    if(random_particles[1] >= random_particles[0])
        random_particles[1]++;
    
    updatePop(f,&random_particles[0],myData,cost,localState);
}


/// Component Optimization Algorithm: Whale Optimization Algorithm WOA
/// @param myData Vector array of data of individual
/// @param cost Cost of myData for the given function
/// @param localState MTGP32 PRG state
__device__ void woa(int f,float * __restrict__ myData,float * __restrict__ myCost,curandStateMtgp32 *localState,
    int current_iter,int * __restrict__ indexBest, float bound_low, float bound_high){
    
    /// Decreases linearly from 2 to 0
    const float a_1 = 2.0 * __fdividef((max_iter  - current_iter),max_iter);

    /// Decreases linearly from -1 to -2
    const float a_2 = (-1.0 * __fdividef((max_iter  - current_iter), max_iter)) - 1.0;

    float l = (a_2 - 1)* curand_uniform(localState) + 1;
    
    float beta = curand_uniform(localState);
    

    /// Pick the random individual
    int index[2];
    index[0]=*indexBest;
    index[1] = int(curand_uniform(localState) * (psize-1))-1;
    if(index[0] >= threadIdx.x)
        index[1]++;

    /// The variables for values that change according to predicate
    float a[2],c[2],d,r;

    // Remeber to check the random variable distribution bounds for index
    r = curand_uniform(localState);
    c[0] = 2.0 * r;
    c[1] = 1;
    a[0] = -(2.0 * a_1 * r - a_1);
    a[1] = __expf(b * l) * __cosf( 2.0 * PI * l);
    
    /// p: 0 or 1
    const int p=beta >= 0.5;
    /// alpha 0 when p == 0 && (abs(a) < 1)
    const int alpha = !p && (fabsf(a[0]) >= 1);

    /// The variables for other individual which can be the best or random
    float data_p[dimension],cost_p;

    /// Get the data from the other threads in the warp

    getData(index[alpha],myData,myCost,data_p,&cost_p);

    for (int j = 0; j < dimension; j++) {
        d = fabsf(c[p] * data_p[j] - myData[j]);

        myData[j] = data_p[j] + a[p] * d;
        /// check solution bound
        if (myData[j] < bound_low){
            myData[j] = bound_low;
        } else if (myData[j] > bound_high){
            myData[j] = bound_high;
        }
    }

    func(f,myData,myCost);
}