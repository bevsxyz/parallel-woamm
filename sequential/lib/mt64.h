/// @file mt64.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief RNG
/// @version 1.0
/// @date 2021-05-06

#ifndef mt64_h
#define mt64_h

#include <random>

/// An object class for RNG
class randomUniform{
    public:
        explicit randomUniform() 
        : gen(rd()),uniform_real(0, 1){}

        /// Reset Seed 
        void set_seed(){
            gen.seed(rd());
        }

        /// Returns Uniform Integer
        int integer(){
            return uniform_integer(gen);
        }

        /// Returns Unform Real
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