#include <iostream>
#include <vector>

#include "runner.h"
#include "./func/func.h"
#include "./data/data_stats.h"

using namespace std;

int main() {
    int experiment = 50;
    vector<DataStats> result_best;
 
    result_best.push_back(runFunc(experiment, "sphere", func::sphere, -100, 100)); //
    result_best.push_back(runFunc(experiment, "rosenbrock", func::rosenbrock, -30, 30));
    result_best.push_back(runFunc(experiment, "rastrigin", func::rastrigin, -5.12, 5.12));
    result_best.push_back(runFunc(experiment, "griewangk", func::griewangk, -600, 600));

    output_all(result_best);
    cout << "Finished!\n";
    return 0;
}
