/// @file runner.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief runner
/// @version 1.0
/// @date 2021-05-08

#ifndef runner_h
#define runner_h

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include <vector>
#include "./data/data_stats.h"
#include "./oa/oa.h"

using namespace std;

DataStats runFunc(int experiment, string func_name,int, float l, float u);

void output_func(string func_name, DataStats result, vector<vector<float>> f_bests_history);

void output_all(vector<DataStats> result_best);

#endif //run_h