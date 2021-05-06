/// @file mt64.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief RNG
/// @version 1.0
/// @date 2021-05-06

#ifndef mt64_h
#define mt64_h

#include <random>

class randomUniform{
    public:
        explicit randomUniform(int lower_bound, int upper_bound) 
        : gen(rd()),uniform_real(lower_bound, upper_bound){}
        void set_seed(){
            gen.seed(rd());
        }
        int integer(int lower_bound, int upper_bound){
            uniform_integer.param(std::uniform_int_distribution<int>::param_type(lower_bound, upper_bound));
            return uniform_integer(gen);
        }
        double real(){
            return uniform_real(gen);
        }
    private:
        std::random_device rd;
        std::mt19937_64 gen;
        std::uniform_int_distribution<> uniform_integer;
        std::uniform_real_distribution<> uniform_real;
};

#endif /* mt64_hpp */