#ifndef WARP_STATUS_CUH
#define WARP_STATUS_CUH


namespace Themis{
    
    struct WarpsStatus {

        unsigned int data;
        
        __forceinline__ __device__ unsigned int addTotalWarpNum(unsigned int a) {
            unsigned int prev = atomicAdd(&data, a);
            return (0xFFFF & prev);
        }

        __forceinline__ __device__ unsigned int addIdleWarpNum() {
            unsigned int prev = atomicAdd(&data, 0x00010000u);
            return (unsigned int) (0x7FFF & (prev >> 16));  
        }

        __forceinline__ __device__ unsigned int addIdleWarpNum(unsigned int &num_warps) {
            unsigned int prev = atomicAdd(&data, 0x00010000u);
            num_warps = (unsigned int) (0xFFFF & prev);
            return (unsigned int) (0x7FFF & (prev >> 16));  
        }

        __forceinline__ __device__ unsigned int subIdleWarpNum() {
            unsigned int prev = atomicSub(&data, 0x00010000u);
            return (unsigned int) (0x7FFF & (prev >> 16));  
        }

        __forceinline__ __device__ void fetchWarpNum(unsigned int &num_idle_warps, unsigned int &num_warps) {
            unsigned int prev = atomicCAS(&data, 0xFFFFFFFFu,  0xFFFFFFFFu);
            num_idle_warps = 0x7FFF & (unsigned int) (prev >> 16);
            num_warps = (unsigned int) (0xFFFF & prev);
        }

        __forceinline__ __device__ bool isTerminated() {
            unsigned int prev = atomicCAS(&data, 0xFFFFFFFFu, 0xFFFFFFFFu);
            return prev >> 31; 
        }

        __forceinline__ __device__ void terminate() {
            atomicAdd(&data, 1u << 31);
        }
    };
}

#endif