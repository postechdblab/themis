#ifndef __PUSHEDPARTS_CUH__
#define __PUSHEDPARTS_CUH__


namespace Themis{ namespace PushedParts {
    
    struct PushedPartsAtZeroLvl {
        volatile int lvl;
        volatile int size;
        volatile int cnt;
        volatile int start;
        volatile int end;
    };

    struct PushedPartsAtLoopLvl {
        volatile int lvl;
        volatile int size;
        volatile int cnt;
        volatile int start[64];
        volatile int end[64];
        __forceinline__ __device__ char* GetAttrsPtr() { return ((char*) this) + sizeof(PushedPartsAtLoopLvl); }    
    };

    struct PushedPartsAtIfLvl {
        volatile int lvl;
        volatile int size;
        volatile int cnt;
        __forceinline__ __device__ char* GetAttrsPtr() { return ((char*) this) + sizeof(PushedPartsAtIfLvl); }    
    };

    struct PushedParts {
        volatile int lvl;
        volatile int size;
    };
    
    struct PushedPartsStack {
        int lock;
        volatile int num;
        volatile int start;


        __forceinline__ __device__ bool isLockFree() {
            return 0 == atomicCAS(&lock, -1, -1);
        }

        __forceinline__ __device__ bool TryLock() {
            return 0 == atomicCAS(&lock, 0, 1);
        }

        __forceinline__ __device__ void FreeLock() {
            atomicCAS(&lock, 1, 0);
        }

        __forceinline__ __device__ char* PushPartsAtLoopLvl(int _lvl, int _size) {
            start += sizeof(PushedPartsAtLoopLvl) + _size;
            num += 1;
            return ((char*) this) - start;
        }

        __forceinline__ __device__ char* PushPartsAtIfLvl(int _lvl, int _size) {
            start += sizeof(PushedPartsAtIfLvl) + _size;
            num += 1;
            return ((char*) this) - start;
        }

        __forceinline__ __device__ char* PushPartsAtZeroLvl() {
            start += sizeof(PushedPartsAtZeroLvl);
            num += 1;
            return ((char*) this) - start;
        }

        __forceinline__ __device__ PushedParts* Top() { //int warp_id, int thread_id) {
            int s = start;
            //printf("Warp %d %d %d 4\n", warp_id, thread_id, s);
            //s = __shfl_sync(ALL_LANES, s, 0);
            //printf("Warp %d %d 5, %d\n", warp_id, thread_id, s);
            return (PushedParts*) (((char*) this) - s);
        }

        __forceinline__ __device__ void PopPartsAtLoopLvl(int _size) {
            start -= sizeof(PushedPartsAtLoopLvl) + _size;
            num -= 1;
        }

        __forceinline__ __device__ void PopPartsAtIfLvl(int _size) {
            start -= sizeof(PushedPartsAtIfLvl) + _size;
            num -= 1;
        }

        __forceinline__ __device__ void PopPartsAtZeroLvl() {
            start -= sizeof(PushedPartsAtZeroLvl);
            num -= 1;
        }

    };

    __forceinline__ __device__ PushedPartsStack* GetIthStack(PushedPartsStack* stack, 
        size_t size_per_warp, size_t i) 
    {
        return (PushedPartsStack*) (((char*) stack) + size_per_warp * i);
    }

    __global__ void Krnl_InitPushedPartsStack(PushedPartsStack* stack, size_t size_per_warp, size_t num_warps) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_warps) return;
        stack = (PushedPartsStack*) (((char*) stack) + size_per_warp * tid);
        stack->lock = 0;
        stack->num = 0;
        stack->start = 0;
    }


    void InitPushedPartsStack(PushedPartsStack* &stack, size_t &size_per_warp, int64_t size, int num_warps) {

        printf("Initialize pushed parts stack for %d warps\n", num_warps);
        size_per_warp = 1024 * 32;
        char* result = NULL;
        //num_warps = 82 * 64;
        std::cout << "Init pushed parts stack " << size_per_warp << std::endl;
        cudaMalloc((void**)&result, size_per_warp * (num_warps));
        stack = (PushedPartsStack*) (result + size_per_warp - 128);
        Krnl_InitPushedPartsStack<<<1024,128>>>(stack, size_per_warp, (size_t) num_warps);
    }

}}


#endif