#include <iostream>
#include <vector>

#include "runner.h"


#include "./data/data_stats.h"

using namespace std;

int main() {
    int experiment = 50;
    vector<DataStats> result_best;
 
    result_best.push_back(runFunc(experiment, "sphere", 1, -100, 100)); //
    result_best.push_back(runFunc(experiment, "rosenbrock", 2, -30, 30));
    result_best.push_back(runFunc(experiment, "rastrigin", 3, -5.12, 5.12));
    result_best.push_back(runFunc(experiment, "griewangk", 4, -600, 600));

    output_all(result_best);
    cout << "Finished!\n";
    return 0;
}
