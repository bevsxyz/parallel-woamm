/// @file mpgp32.h
/// @author Bevan Stanely (bevanstanely@iisc.ac.in)
/// @brief RNG
/// @version 1.0
/// @date 2021-05-14

#ifndef mpgp32_h
#define mpgp32_h

#include <curand.h>
#include <curand_kernel.h>
/// include MTGP host helper functions
#include <curand_mtgp32_host.h>
/// include MTGP pre-computed parameter sets
#include <curand_mtgp32dc_p_11213.h>

class cudarandoms {
    public:
        curandStateMtgp32 *devMTGPStates;      /// State array for MTGP32 generator
        mtgp32_kernel_params *devKernelParams; /// Parameters for initialising PRG
        cudarandoms(){
            /// Allocate space for prng states on device
            cudaMalloc((void **)&devMTGPStates, 32 * sizeof(curandStateMtgp32));
            
            /// Allocate space for MTGP kernel parameters
            cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));

            /// Reformat from predefined parameter sets to kernel format,
            /// and copy kernel parameters to device memory
            curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
            
            /// Initialize one state per thread block
            curandMakeMTGP32KernelState(devMTGPStates, 
                        mtgp32dc_params_fast_11213, devKernelParams, 32, time(NULL));
        }
};

namespace myrandom {
    cudarandoms die;
}

#endif /* mpgp32_h */