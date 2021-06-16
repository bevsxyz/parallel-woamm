#include <iostream>
#include <cmath>
#include <vector>

#include "../lib/mt64.h"
#include "runner.h"
#include "./func/func.h"
#include "./data/data_stats.h"

using namespace std;

int main(int argc, char * argv[]) {
    randomUniform rnd_global;
    int experiment = 50;
    vector<DataStats> result_best;

    int iterations = atoi(argv[1]);
 
    result_best.push_back(runFunc(experiment, "sphere", func::sphere, -100, 100, iterations,&rnd_global)); //
    result_best.push_back(runFunc(experiment, "rosenbrock", func::rosenbrock, -30, 30, iterations,&rnd_global));
    result_best.push_back(runFunc(experiment, "rastrigin", func::rastrigin, -5.12, 5.12, iterations,&rnd_global));
    result_best.push_back(runFunc(experiment, "griewangk", func::griewangk, -600, 600, iterations,&rnd_global));

    output_all(result_best);
    cout << "Finished!\n";
    return 0;
}
