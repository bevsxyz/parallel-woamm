/// @file runner.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief runner
/// @version 1.0
/// @date 2021-05-08

#ifndef runner_h
#define runner_h

#include <vector>
#include "./data/data_stats.h"
#include "./whale/woam.h"
#include "../lib/mt64.h"

using namespace std;

DataStats runFunc(int experiment, string func_name, float (*f)(vector<float> &), float l, 
float u,int iterations, randomUniform *rnd_global);

void output_func(string func_name, DataStats result, vector<vector<float>> f_bests_history);

void output_all(vector<DataStats> result_best);

#endif //run_h