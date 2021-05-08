#include "woam.h"
#include "../population/population.h"
#include "../../lib/mt64.h"
#include <iostream>
#include <cmath>

using namespace std;
WOAM::WOAM(float (*f)(vector<float> &), float l, float u,  randomUniform *rnd_global){
    function = f;
    bound_low = l;
    bound_high = u;
    population.init(psize, dimension, f, l, u);
    rnd = rnd_global;
}

/// Calculate the fitness for all individuals
/// and update global fitness
void WOAM::get_best_fitness(){
    for(int i = 0;i <psize;i++){
        /// get fitness
        population.cost[i] = function(population.data[i]); 
    }
    /// update global best
    population.get_best();
}

/// Update the population data
/// @param i The current individual
/// @param x Index of individual with lower fitness
/// @param y Index of individual with higher fitness
void WOAM::msos_update_pop(int i, int x, int y){
    int bf1,bf2;
    float mv,temp,cost_i,cost_y;
    vector<float> p_i, p_y;
    bf1 = 1 + rnd->real()*2 ;
    bf2 = 1 + rnd->real()*2 ;
    for (int j = 0; j < dimension; j++){
        mv = (population.data[i][j]+population.data[y][j])/2;
        temp = population.data[i][j] + (rnd->real()* (population.data[x][j]-mv*bf1));
        p_i.push_back(temp);
        temp = population.data[y][j] + (rnd->real()* (population.data[x][j]-mv*bf2));
        p_y.push_back(temp);
    }
    cost_i = function(p_i);
    if(cost_i < population.cost[i]){
        population.cost[i] = cost_i;
        for (int j = 0; j < dimension; j++){
            population.data[i][j] = p_i[j];
        }
    }
    cost_y = function(p_y);
    if(cost_y < population.cost[y]){
        population.cost[y] = cost_y;
        for (int j = 0; j < dimension; j++){
            population.data[y][j] = p_y[j];
        }
    }
}

/// Modified Mutualism Phase of SOS
void WOAM::msos(){
    int m, n,bf1,bf2;
    float mv,temp;
    for (int i = 0; i < psize; i++){
        m = int(rnd->real() * (psize-1));
        if(m >= i)
            m++;
        n = int(rnd->real() * (psize-2));
        if(n >= i)
            n++;
        if(n >= m)
            n++;
        if(population.cost[m]<population.cost[n]){
            msos_update_pop(i,m,n);
        }
        else{
            msos_update_pop(i,n,m);
        }
    }
    population.get_best();
}

/// Whale Optimization Algorithm Component
/// @param k current iteration
void WOAM::woa(int k){
    float a_1 = 2.0 * (max_iter  - k)/ max_iter;  /// Decreases linearly from 2 to 0
    float a_2 = (-1.0 * (max_iter  - k)/ max_iter) - 1.0; /// Decreases linearly from -1 to -2

    for (int i = 0; i < psize; i++){
        float beta = rnd->real();
        int particle_rand = int(rnd->real() * psize);
        float l = (a_2 - 1)* rnd->real() + 1;

        for (int j = 0; j < dimension; j++) {
            if (beta < 0.5) {
                float r = rnd->real();
                float a = 2.0 * a_1 * r - a_1;
                float c = 2.0 * r;
                float d;

                if (fabs(a) < 1) {
                    d = fabs(c * population.data_best[j] - population.data[i][j]);
                    population.data[i][j] = population.data_best[j] - a * d;
                } else {
                    d = fabs(c * population.data[particle_rand][j] - population.data[i][j]);
                    population.data[i][j] = population.data[particle_rand][j] -  a * d;
                }
            } else {
                float d = fabs(population.data_best[j] - population.data[i][j]);
                population.data[i][j] = d * pow(M_E, b * l) * cos( 2.0 * M_PI * l) +  population.data_best[j];
            }
            /// check solution bound
            if (population.data[i][j] < bound_low){
                population.data[i][j] = bound_low;
            } else if (population.data[i][j] > bound_high){
                population.data[i][j] = bound_high;
            }
        }

        get_best_fitness();
    }
}

vector<float> WOAM::run(){
    population.reset(rnd);
    vector<float> global_best_history;

    get_best_fitness();

    for (int k = 0; k < max_iter; k++){

        // mMSOS Component
        msos();

        // WOA Component
        woa(k);

        global_best_history.push_back(population.cost_best);
    }
    return global_best_history;
}