/// @file woam.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Enhanced Whale Optimization Algorithm (WOAmM)
/// @version 1.0
/// @date 2021-05-07


#ifndef woam
#define woam

#include "../population/population.h"
#include "../data/data_stats.h"
#include "../../lib/mt64.h"

using namespace std;

class WOAM{
public:
    WOAM(float (*f)(vector<float> &), float l, float u, randomUniform *rnd_global);
    vector<float> run();

private:
    int max_iter = 30; /// Max iteration
    float b = 1;
    int psize = 32; /// Size of population
    int dimension = 30; /// Dimension
    Population population;

    randomUniform *rnd;
    float (*function)(vector<float> &); /// Objective function
    float bound_low; /// Objective function solution lower bound
    float bound_high; /// Objective function solution higher bound

    void get_best_fitness(); /// Calculate the best fitness
    void msos_update_pop(int i, int x, int y);
    void msos(); /// Modified Mutualism Phase of SOS
    void woa(int k); /// Whale Optimisation Component
};
#endif //woam