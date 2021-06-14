#include "runner.h"
#define blocks 1

using namespace std;

/// run woam for a function
/// @param experiment number of experiment
/// @param f function
/// @param l low x bound
/// @param u up x bound
/// @return return result analysis
DataStats runFunc(int experiment, string func_name, int f, float l, float u){
    DataStats result;
    result.func_name = func_name;
    vector<vector<float>> f_bests_history;
    float time_temp, alloc_time;
    float *device_solution;


    chrono::high_resolution_clock::time_point start_alloc = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&device_solution, blocks * sizeof(float));
    chrono::high_resolution_clock::time_point finish_alloc = chrono::high_resolution_clock::now();
    alloc_time = chrono::duration_cast<chrono::microseconds>(finish_alloc - start_alloc).count();


    for (int i = 0; i < experiment; i++){
        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
        vector<float> f_best_history = run(f, l, u,device_solution);
        chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
        float f_best = f_best_history[f_best_history.size()-1];
        result.data.push_back(f_best);
        f_bests_history.push_back(f_best_history);

        time_temp = chrono::duration_cast<chrono::microseconds>(finish - start).count();
        result.time.push_back(time_temp);
    }

    start_alloc = chrono::high_resolution_clock::now();
    cudaFree(device_solution);
    finish_alloc = chrono::high_resolution_clock::now();
    alloc_time += chrono::duration_cast<chrono::microseconds>(finish_alloc - start_alloc).count();
    
    if(f == 1){
        ofstream file("./out/memory_alloc_dealloc_time.csv",std::ios_base::app);
        file << alloc_time << endl;
        file.close(); 
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