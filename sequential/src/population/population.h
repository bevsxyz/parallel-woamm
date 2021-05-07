/// @file population.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief Initialize Population
/// @version 1.0
/// @date 2021-05-07

#ifndef population_h
#define population_h

#include <stdio.h>
#include <vector>

using namespace std;

/// Population
class Population{
public:
    void init(int s, int d, float (*f)(vector<float> &), float l, float u);

    vector<vector<float>> data;
    vector<float> cost;

    vector<float> data_best;
    float cost_best;

    void reset(randomUniform *rnd); /// Reset the population for new run
    void get_best();

private:
    int size;
    int dimension;
    int bound_low;
    int bound_up;
    float (*function)(vector<float> &);

    void generate(randomUniform *rnd);
};

#endif /* population_hpp */
