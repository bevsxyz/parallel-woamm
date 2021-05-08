#include "func.h"
#include <cmath>
#include <vector>

using namespace std;

/// Function 1, Implementation of Sphere function
/// @param x descriptionx Vector of float
float func::sphere(vector<float> &x){
    float result = 0;
    for (int i = 0; i < x.size(); i++){
        result += x[i] * x[i];
    }
    return result;
}

/// Function 2, Implementation of Rosenbrock's function
/// @param x descriptionx Vector of float
float func::rosenbrock(vector<float> &x){
    float result = 0;
    for (int i = 0; i < x.size() - 1; i++){
        float a = x[i] * x[i] - x[i+1];
        float b = 1.0 - x[i];
        result += 100 * a * a + b * b;
    }
    return result;
}

/// Function 3, Implementation of Rastrigin's function
/// @param x descriptionx Vector of float
float func::rastrigin(vector<float> &x){
    float result = 0;
    for (int i = 0; i < x.size() ; i++){
        result += x[i] * x[i] - 10.0 * cos((2 * M_PI * x[i]));
    }
    return 10 * x.size() + result;
}

/// Function 4, Implementation of Griewangk function
/// @param x descriptionx Vector of float
float func::griewangk(vector<float> &x){
    float sum = 0;
    float product = 1;
    for (int i = 0; i < x.size() ; i++){
        sum += x[i] * x[i] / 4000;
        product *= cos(x[i]/sqrt(i+1));
    }
    return 1.0 + sum - product;
}