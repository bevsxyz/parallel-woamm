/// @file philox4.cu
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Parallel Implementation of WOAmM with Philox 4x32 10 RNG
/// @version 1.0
/// @date 2021-06-16

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cmath>
#include <bits/stdc++.h>

#include <curand.h>
#include <curand_kernel.h>

#define fpsize 30
#define PI 3.14159265358979323846
#define psize 32
#define dimension 30


using namespace std;

/// Functions
//-------------------------------------------------------------------------------------------------------

// Variable dimensions unimodal functions

__device__ float sphere(const float * __restrict__ x);
__device__ float rosenbrock(const float * __restrict__ x);

// Variable dimensions multimodal functions

__device__ float rastrigin(const float * __restrict__ x);
__device__ float griewangk(const float * __restrict__ x);

__device__ void func(int f, const float * __restrict__ mydata,float * __restrict__ cost);

/// WOAmM functions for GPU
//-------------------------------------------------------------------------------------------------------

vector<float> run(int f, float l, float u,float *device_solution,int blocks,int max_iter,curandStatePhilox4_32_10_t * __restrict__ global_state);    
__global__ void woam(const int f,const float l, const float u, float*solution,const int max_iter,curandStatePhilox4_32_10_t * __restrict__ global_state);

__device__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest);

__device__ void updatePop(const int myID,const int f,
    const int * __restrict__ random_particles,float * __restrict__ my_Data,
    float * __restrict__ my_cost, curandStatePhilox4_32_10_t * __restrict__ localState);

__device__ void msos(const int my_index,const int f,float * __restrict__ myData,
    float * __restrict__ cost,curandStatePhilox4_32_10_t * __restrict__ localState);

__device__ void woa(const int myID,const int f,float * __restrict__ myData,
    float * __restrict__ cost,curandStatePhilox4_32_10_t * __restrict__ localState,
    const int current_iter,int * __restrict__ indexBest,
     const float bound_low, const float bound_high,const int max_iter);

/// Data statistics specific data structure and functions
//-------------------------------------------------------------------------------------------------------

class DataStats
{
public:
    float mean;
    float median;
    float stand;
    float range[2];
    float time_avg;
    string func_name;
    vector<float> time;
    vector<float> data;

    void run();

private:
    void get_mean();
    void get_median();
    void get_stand();
    void get_range();
    void get_timeavg();
};


/// Runner function for WOAmM
//-------------------------------------------------------------------------------------------------------

DataStats runFunc(int experiment, string func_name,int, float l, float u,int blocks,int max_iter,curandStatePhilox4_32_10_t *  __restrict__ global_state,float * device_solution);

void output_func(string func_name, DataStats result, vector<vector<float>> f_bests_history);

void output_all(vector<DataStats> result_best);

//-------------------------------------------------------------------------------------------------------


/// GPU kernel to initialise the RNG
__global__ void setup_kernel(int seed,curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x+(32*blockIdx.x);
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, /* the seed controls the sequence of random values that are produced */
    id, /* the sequence number is only important with multiple cores */
    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
    &state[id]);
}

int main(int argc, char *argv[]) {
    int experiment = 50;// Each function is ran 50 times
    vector<DataStats> result_best;

    /// The number of blocks and maximum no. of iterations for  WOAmM are commanline arguments
    int blocks = atoi(argv[1]);
    int max_iter = atoi(argv[2]);

    float alloc_time;
    float *device_solution;

    curandStatePhilox4_32_10_t *global_state;


    /// We keep track of the allocation times for RNG and float
    chrono::high_resolution_clock::time_point start_alloc = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&device_solution, blocks * sizeof(float));
    cudaMalloc((void**)&global_state, blocks * 32 * sizeof(curandStatePhilox4_32_10_t));
    setup_kernel<<<blocks,psize>>>(time(NULL),global_state);
    chrono::high_resolution_clock::time_point finish_alloc = chrono::high_resolution_clock::now();
    alloc_time = chrono::duration_cast<chrono::microseconds>(finish_alloc - start_alloc).count();
    //---------------------------------------------------------------------------------------------

    result_best.push_back(runFunc(experiment, "sphere", 1, -100, 100,blocks,max_iter,global_state,device_solution));
    result_best.push_back(runFunc(experiment, "rosenbrock", 2, -30, 30,blocks,max_iter,global_state,device_solution));
    result_best.push_back(runFunc(experiment, "rastrigin", 3, -5.12, 5.12,blocks,max_iter,global_state,device_solution));
    result_best.push_back(runFunc(experiment, "griewangk", 4, -600, 600,blocks,max_iter,global_state,device_solution));

    // Along with deallocation times
    start_alloc = chrono::high_resolution_clock::now();
    cudaFree(device_solution);
    cudaFree(global_state);
    finish_alloc = chrono::high_resolution_clock::now();
    alloc_time += chrono::duration_cast<chrono::microseconds>(finish_alloc - start_alloc).count();
    //---------------------------------------------------------------------------------------------
    
    // Save the allocation + deallocation times to a file
    ofstream file("./out/memory_alloc_dealloc_time.csv");
    file << alloc_time << endl;
    file.close(); 
    //---------------------------------------------------------------------------------------------

    output_all(result_best);
    cout << "Finished!\n";
    return 0;
}

/// Functions
//-------------------------------------------------------------------------------------------------------

/// The main function which chooses the function to be optimized at runtime
/// Returns the cost
/// @param f the nummber code of the function to be optimized
/// @param mydata The position vector & input for the function
/// @param cost The output and fitness value
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
}

/// Function 1, Implementation of Sphere function
/// @param x Position vector of float
__device__ float sphere(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize; i++){
        result += x[i] * x[i];
    }
    return result;
}

/// Function 2, Implementation of Rosenbrock's function
/// @param x Position vector of float
__device__ float rosenbrock(const float * __restrict__ x){
    float a[3];
    a[2] = 0;
    for (int i = 0; i < fpsize - 1; i++){
        a[0] = x[i] * x[i] - x[i+1];
        a[1] = 1.0 - x[i];
        a[2] += 100 * a[0] * a[0] + a[1] * a[1];
    }
    return a[2];
}

/// Function 3, Implementation of Rastrigin's function
/// @param x Position vector of float
__device__ float rastrigin(const float * __restrict__ x){
    float result = 0;
    for (int i = 0; i < fpsize ; i++){
        result += x[i] * x[i] - 10.0 * __cosf(2 * PI * x[i]);
    }
    return 10 * fpsize + result;
}

/// Function 4, Implementation of Griewangk function
/// @param x Position vector of float
__device__ float griewangk(const float * __restrict__ x){
    float sum = 0;
    float product = 1;
    float rec = 1 / 4000;
    for (int i = 0; i < fpsize ; i++){
        sum += x[i] * x[i] * rec;
        product *= __cosf(x[i]*rsqrtf(i+1));
    }
    return 1.0 + sum - product;
}

/// WOAmM functions for GPU
//-------------------------------------------------------------------------------------------------------

/// The main function to facilitiate memory copy and kernel calls
/// @param f Function to be evaluvated
/// @param l lower bound of the function
/// @param u upper bound of the function
/// @param device_solution The device array for solutions that is being reused
/// @param blocks the number of blocks to be run
/// @param max_iter The maximum number of iterations
/// @param global_state The global state of RNG
vector<float> run(int f, float l, float u,float *device_solution,int blocks,
    int max_iter,curandStatePhilox4_32_10_t *  __restrict__ global_state){
    vector<float> global_best_solution;

    float host_solution[blocks];

    woam<<<blocks,psize>>>(f,l,u, device_solution,max_iter, global_state);

    cudaMemcpy(host_solution, device_solution, blocks* (sizeof(float)), cudaMemcpyDeviceToHost);
    
    float best = host_solution[0];
    for(int i = 1; i < blocks; i++){
        if(host_solution[i]<best)
            best = host_solution[i];
    }
    global_best_solution.push_back(best);
    
    return global_best_solution;
}

/// The main function for WOAM
/// @param f Function to be evaluvated
/// @param l lower bound of the function
/// @param u upper bound of the function
/// @param solution The solution array to write the solution
/// @param max_iter The maximum number of iterations
/// @param global_state The global state of RNG
__global__ void woam(const int f,const float l, const float u, float*solution,
    const int max_iter,curandStatePhilox4_32_10_t *  __restrict__ global_state){
    const int myID = threadIdx.x;
    const int bID  = blockIdx.x;

    /// Copy state to local memory
    curandStatePhilox4_32_10_t localState = global_state[myID+(32*bID)];
    float myData[dimension],cost,costBest;
    int indexBest=myID;
    for (int j = 0; j < dimension; j++){
        /// generate data
        myData[j] = (float)(l + (curand_uniform(&localState)* (u - l)));
    }
    func(f,&myData[0],&cost);

    for (int k = 0; k < max_iter; k++){
        // mMSOS Component
        msos(myID,f,&myData[0],&cost,&localState);

        costBest = cost;
        indexBest=myID;
        getBest(&indexBest,&costBest);

        // WOA Component
        woa(myID,f,&myData[0],&cost,&localState,k,&indexBest, l,u,max_iter);

    }
    costBest = cost;
    indexBest=myID;

    /// Write state back to global memory
    global_state[myID+(32*bID)] = localState;
    getBest(&indexBest,&costBest);
    if(myID==0)
        solution[bID] = costBest;
}

/// Finds the best cost in the population
/// Uses a butterfly reduction
/// @param indexBest Index of the individual with minimum cost, will be updated
/// @param costBest Cost of the individual with minimum cost, will be updated
__device__ __forceinline__ void getBest(int * __restrict__ indexBest,float * __restrict__ costBest){\
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

/// Update the population for msos
/// @param random_particles Array of the two random individuals picked
/// @param myData Pointer for my data
/// @param myCost Pointer for my cost
/// @param localState RNG local state
__device__ void updatePop(const int myID,const int f,const int * __restrict__ random_particles,float * __restrict__ my_Data,
    float * __restrict__ my_cost, curandStatePhilox4_32_10_t *  __restrict__ localState){
    
    float cost_rp[2],data_rp[dimension];

    cost_rp[0] = __shfl_sync(0xffffffff,*my_cost ,random_particles[0]);
    cost_rp[1] = __shfl_sync(0xffffffff,*my_cost ,random_particles[1]);

    /// Calculate Fitness
    int hIndex , lIndex;

    hIndex = cost_rp[0] < cost_rp[1];
    lIndex = !hIndex;

    float highFitness, lowFitness;

    float my_Data_kp1[dimension];
    float my_cost_kp1;
    
    int bf1,bf2;
    float mv;

    bf1 = 1 + curand_uniform(localState) * 2;
    bf2 = 1 + curand_uniform(localState) * 2;

    /// Calculate the k+1 population values
    for (int i = 0; i < dimension; i++){
        highFitness = __shfl_sync(0xffffffff, my_Data[i],random_particles[hIndex]);
        lowFitness = __shfl_sync(0xffffffff, my_Data[i],random_particles[lIndex]);
        mv = __fdividef((my_Data[i] + highFitness),2);
        my_Data_kp1[i] = my_Data[i] + (curand_uniform(localState) * (lowFitness - mv*bf1));
        data_rp[i] = highFitness  + (curand_uniform(localState) * (lowFitness - mv*bf2));
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
    int index=myID,indexTemp;
    float costTemp;
    for(int i = 0; i < 32; i++){
        costTemp = __shfl_sync(0xffffffff, cost_rp[0],i);
        indexTemp = __shfl_sync(0xffffffff, random_particles[hIndex],i);
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
/// @param localState RNG local state
__device__ void msos(const int my_index,const int f,float * __restrict__ myData,
    float * __restrict__ cost,curandStatePhilox4_32_10_t *  __restrict__ localState){
    int random_particles[2];
    random_particles[0] = int(curand_uniform(localState) * (psize-1))-1;
    if(random_particles[0] >= my_index)
        random_particles[0]++;
    random_particles[1] = int(curand_uniform(localState) * (psize-2))-1;
    if(random_particles[1] >= my_index)
        random_particles[1]++;
    if(random_particles[1] >= random_particles[0])
        random_particles[1]++;
    
    updatePop(my_index,f,&random_particles[0],myData,cost,localState);
}


/// Component Optimization Algorithm: Whale Optimization Algorithm WOA
/// @param myData Vector array of data of individual
/// @param cost Cost of myData for the given function
/// @param localState RNG local state
/// @param current_iter The current iteration count
/// @param indexBest The index of the best individual
/// @param bound_low Lower bound of function
/// @param bound_high Upper bound of function
/// @param max_iter The maximum number of iterations
__device__ void woa(const int myID,const int f,float * __restrict__ myData,float * __restrict__ myCost,
    curandStatePhilox4_32_10_t *  __restrict__ localState,const int current_iter,
    int * __restrict__ indexBest, const float bound_low, const float  bound_high,const int max_iter){
    
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
    if(index[0] >= myID)
        index[1]++;

    /// The variables for values that change according to predicate
    float a[2],c[2],d,r;

    // Remeber to check the random variable distribution bounds for index
    
    
    /// p: 0 or 1
    const int p=beta >= 0.5;
    int alpha;
    /// The variables for other individual which can be the best or random
    float data_p;

    c[1] = 1;
    /// b == 1 so just using the literal value
    a[1] = __expf(1 * l) * __cosf( 2.0 * PI * l);

    for (int j = 0; j < dimension; j++) {
        r = curand_uniform(localState);
        c[0] = 2.0 * r;
        a[0] = -(2.0 * a_1 * r - a_1);
        
        /// alpha 0 when p == 0 && (abs(a) < 1)
        alpha = !p && (fabsf(a[0]) >= 1);

        /// Fetch the jth position data of the individual
        data_p = __shfl_sync(0xffffffff, myData[j],index[alpha]);
        
        d = fabsf(c[p] * data_p - myData[j]);

        myData[j] = data_p + a[p] * d;

        /// check solution bound
        if (myData[j] < bound_low){
            myData[j] = bound_low;
        }
        if (myData[j] > bound_high){
            myData[j] = bound_high;
        }
    }

    /// Evaluvate the function i.e find fitness
    func(f,myData,myCost);
}

/// Runner function for WOAmM
//-------------------------------------------------------------------------------------------------------

/// run woam for a function
/// @param experiment number of experiment
/// @param f function
/// @param l low x bound
/// @param u up x bound
/// @param blocks The number of GPU blocks to call
/// @param max_iter The maximum number of iterations
/// @param global_state The global state of RNG
/// @param device_solution Float array for storing solution in GPU
/// @return return result analysis
DataStats runFunc(int experiment, string func_name, int f, float l, float u,int blocks,
    int max_iter,curandStatePhilox4_32_10_t *  __restrict__ global_state,float * device_solution){
    DataStats result;
    result.func_name = func_name;
    vector<vector<float>> f_bests_history;
    float time_temp;

    for (int i = 0; i < experiment; i++){
        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
        vector<float> f_best_history = run(f, l, u,device_solution,blocks,max_iter, global_state);
        chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
        float f_best = f_best_history[f_best_history.size()-1];
        result.data.push_back(f_best);
        f_bests_history.push_back(f_best_history);

        time_temp = chrono::duration_cast<chrono::microseconds>(finish - start).count();
        result.time.push_back(time_temp);
    }

    result.run();
    output_func(func_name,result,f_bests_history);
    cout << "done: " << func_name << endl;
    return result;
}

/// write result to a file for one function
/// @param func_name function name
/// @param result result stat
/// @param f_bests_history cost history
void output_func(string func_name, DataStats result, vector<vector<float>> f_bests_history){
    /// output result stats
    ofstream file("./out/" + func_name + "_stats.csv");
    file << "Mean,Median,Std,Range(low),Range(high),Time(microseconds)" << endl;
    file << result.mean << "," ;
    file << result.median << "," ;
    file << result.stand << "," ;
    file << result.range[0] << ",";
    file << result.range[1] << "," ;
    file << result.time_avg << endl;
    file.close();

    /// output time history
    ofstream file_timeHistory("./out/"+ func_name + "_timeHistory.csv");
    for (int i = 0; i < result.time.size(); i++){
        if ( i == f_bests_history.size() - 1){
            file_timeHistory << result.time[i] << endl;
        }else{
            file_timeHistory << result.time[i] << ",";
        }
    }
    file_timeHistory.close();

    /// output f history
    ofstream file_fHistory("./out/" + func_name + "_fHistory.csv");
    for (int i = 0; i < f_bests_history.size(); i++){
        for (int j = 0; j < f_bests_history[i].size(); j++) {
            if (j == f_bests_history[i].size() - 1) {
                file_fHistory << f_bests_history[i][j] << endl;
            } else {
                file_fHistory << f_bests_history[i][j] << ",";
            }
        }
    }
    file_fHistory.close();
}

/// write best result for every function
/// @param result_best best result for each function
void output_all(vector<DataStats> result_bests){
    ofstream file("./out/woam_best_stats.csv");
    file << "Function,Name,Mean,Median,Std,Range(low),Range(high),Time(microseconds)" << endl;
    for (int i = 0; i < result_bests.size(); i++){
        file << i + 1 << ",";
        file << result_bests[i].func_name << ",";
        file << result_bests[i].mean << "," ;
        file << result_bests[i].median << ",";
        file << result_bests[i].stand << "," ;
        file << result_bests[i].range[0] << ",";
        file << result_bests[i].range[1] << ",";
        file << result_bests[i].time_avg << endl;
    }
    file.close();
}

/// Data statistics specific data structure and functions
//-------------------------------------------------------------------------------------------------------

/// Generate analytical data
void DataStats::run()
{
    get_mean();
    get_median();
    get_stand();
    get_range();
    get_timeavg();
}

/// Get data data mean
void DataStats::get_mean()
{
    double sum = 0;
    for (int i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    mean = (float)(sum / data.size());
}

/// Get data data median
void DataStats::get_median()
{
    vector<float> data_temp = data;
    sort(data_temp.begin(), data_temp.end());
    median = (data_temp[data_temp.size() / 2 - 1] + data_temp[data_temp.size() / 2]) / 2;
}

/// Get data data standard deviation
void DataStats::get_stand()
{
    double variance = 0.0;
    for (float v : data)
    {   variance += pow(v - mean, 2);
    }
    variance /= data.size();
    stand = (float)(sqrt(variance));
}

/// Get data data range
void DataStats::get_range(){
    float min = data[0];
    float max = data[0];
    for (int i = 1; i < data.size(); i++){
        if (data[i] < min){
            min = data[i];
        }else if (data[i] > max){
            max = data[i];
        }
    }
    range[0] = min;
    range[1] = max;
}

/// Get average time
void DataStats::get_timeavg(){
    double time_temp = 0.0;
    for (float v : time){
        time_temp += v;
    }
    time_avg = (float)(time_temp / data.size());
}
