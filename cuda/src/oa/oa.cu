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

/// The main function for WOAM
__global__ void OA::woam(){
    int myID = threadIdx.x;
    curandStateMtgp32 localState = devMTGPStates[myID];
    float myData[dimension],cost,costBest;
    int indexBest=myID;

    for (int j = 0; j < dimension; j++){
        /// generate data
        myData[j] = (float)(bound_low + (curand_uniform(&localState)* (bound_high - bound_low));
    }
    cost = function(myData);
    costBest = cost;

    getBest(&indexBest,&costBest);

    for (int k = 0; k < max_iter; k++){
        // mMSOS Component
        msos(&myData,&cost,&localState);

        costBest = cost;
        indexBest=myID;
        getBest(&indexBest,&costBest);

        // WOA Component
        woa(&myData,&cost,k,&localState,&indexBest,&costBest);

        costBest = cost;
        indexBest=myID;
        getBest(&indexBest,&costBest);
    }

}

/// Finds the best cost in the population
/// Uses a butterfly reduction
/// @param indexBest Index of the individual with minimum cost, will be updated
/// @param costBest Cost of the individual with minimum cost, will be updated
__device__ void OA::getBest(int * __restrict__ indexBest,float * __restrict__ costBest){
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

/// Update the population for msos
/// @param random_particles Array of the two random individuals picked
/// @param myData Pointer for my data
/// @param myCost Pointer for my cost
/// @param localState MTGP32 PRG state
__device__ void OA::updatePop(const int * __restrict__ random_particles,float * __restrict__ my_Data,
    float * __restrict__ my_cost, curandStateMtgp32 *localState){
    
    float cost_rp[2],data_rp[2*dimension];

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
        mv = (my_Data[x] + data_rp[z])/2;
        my_Data_kp1[x] = my_Data[x] + (curand_uniform(localState) * (data_rp[y] - mv*bf1));
        data_rp[x] = data_rp[z]  + (curand_uniform(localState) * (data_rp[y] - mv*bf2));
    }

    /// Calculate the costs
    my_cost_kp1 = function(my_Data_kp1);
    cost_rp[0] = function(data_rp);

    /// If the new cost is the minima update my individual data
    if(my_cost_kp1 < my_cost){
        my_cost = my_cost_kp1;
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
            if(my_cost<costTemp){
                index = i;
                my_cost = costTemp;
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
__device__ void OA::msos(float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState){
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
    
    updatePop(&random_particles,myData,cost,localState);
}


/// Component Optimization Algorithm: Whale Optimization Algorithm WOA
/// @param myData Vector array of data of individual
/// @param cost Cost of myData for the given function
/// @param localState MTGP32 PRG state
__device__ void OA::woa(float * __restrict__ myData,float * __restrict__ cost,curandStateMtgp32 *localState,
    int current_iter,int * __restrict__ indexBest){
    
    /// Decreases linearly from 2 to 0
    float a_1 = 2.0 * (max_iter  - current_iter)/ max_iter;

    /// Decreases linearly from -1 to -2
    float a_2 = (-1.0 * (max_iter  - current_iter)/ max_iter) - 1.0;

    float l = (a_2 - 1)* curand_uniform(localState) + 1;
    
    float beta = curand_uniform(localState);
    

    /// Pick the random individual
    int index[2];
    index[0]=indexBest;
    index[1] = int(curand_uniform(localState) * (psize-1))-1;
    if(index[0] >= threadIdx.x)
        index[1]++;
    
    /// The arrays for data
    float * particles[2];
    float cost_p[2],data_best[dimension],data_rp[dimension];
    
    /// The pointers are assigned for the respective data
    particles[0] = data_best[0];
    particles[1] = data_rp[0];

    /// Get the data from the other threads in the warp
    getData(index[0],my_Data,my_cost,particles[0],&cost_p[0]);
    getData(index[1],my_Data,my_cost,particles[1],&cost_p[1]);

    /// The variables for values that change according to predicate
    float * d[2];
    float d_vals[3];
    d[0] = d_vals[0];
    d[1] = d_vals[2];
    float a[2],c[2],r;

    // Remeber to check the random variable distribution bounds for index
    int p,alpha;

    for (int j = 0; j < dimension; j++) {
        r = curand_uniform(localState);
        c[0] = 2.0 * r;
        c[1] = 1;

        d[0][0] = fabsf(c[0] * particles[0][j]-myData[j]);
        d[0][1] = fabsf(c[0] * particles[1][j]-myData[j]);
        d[1][0] = fabsf(c[1] * particles[0][j]-myData[j])

        a[0] = -(2.0 * a_1 * r - a_1);
        a[1] = powf(M_E,b * l) * cosf( 2.0 * M_PI * l);

        /// p: 0 or 1
        p = beta >= 0.5;
        /// alpha 0 when p == 0 && (abs(a) < 1)
        alpha = !p && (fabsf(a[0]) >= 1);

        myData[j] = particles[alpha][j] + a[p] * d[p][alpha];
        /// check solution bound
        if (myData[j] < bound_low){
            myData[j] = bound_low;
        } else if (myData[j] > bound_high){
            myData[j] = bound_high;
        }
    }

    cost = function(myData);
}

vector<float> OA::run(){
    vector<float> global_best_solution;

    woam<<1,psize>>(bound_low,bound_high,time(NULL));

    global_best_solution.pushback(*best_solution);
    return global_best_solution;
}