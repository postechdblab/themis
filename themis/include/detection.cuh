#ifndef __TWOLVLBITMAPS_CUH__
#define __TWOLVLBITMAPS_CUH__

#include <stdio.h>
#include <assert.h>
#include "warp_status.cuh"

namespace Themis { namespace Detection {

    __forceinline__ __device__ unsigned int Pick(
        unsigned long long rnd,
        unsigned int num_bits1,
        unsigned long long bits1
    ) {
        unsigned int idx1 = rnd % num_bits1;
        unsigned long long left_bits = idx1 == 0 ? bits1 : bits1 << idx1;
        unsigned long long right_bits = idx1 == 0 ? 0x0ull : bits1 >> (64 - idx1);
        unsigned long long shifted_bits1 = left_bits | right_bits;
        unsigned int idx_bit1 = __ffsll(shifted_bits1) - 1;
        idx_bit1 += idx_bit1 < idx1 ? (num_bits1 - idx1 + 1) : -idx1;
        return idx_bit1;
    }


namespace TwoLvlBitmaps {

    // It assumes only one thread per warp executes this function 
    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Get(
        int warp_id,
        int &tgt_warp_id,
        unsigned long long* bit1, unsigned long long* bit2,
        unsigned int num_idle_warps
        )
    {   
        tgt_warp_id = -1;
        
        if (num_idle_warps == 0) return;

        unsigned long long rnd = (clock64() * 2305843009213693951u + warp_id * 2147483647) % 137438691328u;

        {
            unsigned int idx_bit1 = warp_id / 64;
            bit2 += idx_bit1 * 16;
            unsigned long long sign2 = 0xFull << ((((warp_id % 64)) / 4) * 4);
            unsigned long long bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);
            
            if ((bits2 & sign2) != 0) {
                unsigned int idx_bit2 = Pick(rnd, 64, sign2 & bits2);
                unsigned long long sign2 = 0x1ull << idx_bit2;
                unsigned long long _sign2 = ~sign2;
                unsigned long long sign1 = 0x1ull << idx_bit1;
                do {
                    if ((bits2 & sign2) == 0) return;
                    unsigned long long old_bits2 = bits2;
                    bits2 = atomicCAS(bit2, bits2, _sign2 & bits2);
                    if (bits2 == old_bits2) {
                        tgt_warp_id = idx_bit1 * 64 + idx_bit2;

                        //printf("tgt_warp_id: %d\n", tgt_warp_id);
                        if (bits2 == sign2) {
                            unsigned long long bits1 = atomicCAS(bit1, 0x0ull, 0x0ull);
                            atomicCAS(bit1, bits1, bits1 & (~sign1));
                        }
                        return;
                    }
                    __nanosleep(1<<5);
                } while (true);
                return;
            }
            bit2 -= idx_bit1 * 16;
        }

        const unsigned int num_bits1 = ((NUM_WARPS-1)/64)+1;
        unsigned int idx_bit1 = 0xFFFFFFFF;
        unsigned long long bits1 = 0xFFFFFFFFFFFFFFFFull;

        bits1 = atomicCAS(bit1, 0x0ull, 0x0ull);
        int num_one_in_bits1 = __popc(bits1);

        bool bitmap1_is_used = false;
        if (bits1 && num_idle_warps < (NUM_WARPS / 2)) {
            // Randomly choose the one of one bits
            bitmap1_is_used = true;
            idx_bit1 = Pick(rnd, num_bits1, bits1);
        } 
        if (idx_bit1 == 0xFFFFFFFF) {
            idx_bit1 = rnd % num_bits1;
        }
        bit2 += idx_bit1 * 16;
        unsigned long long bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);
        if (bits2 == 0) {
            unsigned long long sign1 = 0x1ull << idx_bit1;
            atomicCAS(bit1, bits1, bits1 & (~sign1));
            return;
        }
        bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);
        if (bits2 == 0) return;
        
        int num_one_in_bits2 = __popc(bits2);
        unsigned int idx_bit2;
        unsigned int num_bits2 = idx_bit1 == (num_bits1 - 1) ? (NUM_WARPS % 64) : 64;        
        
        if (num_one_in_bits2 >= 32) {
            idx_bit2 = rnd % num_bits2;
        } else {
            idx_bit2 = Pick(rnd, num_bits2, bits2);
        }
        
        unsigned long long sign2 = 0x1ull << idx_bit2;
        unsigned long long _sign2 = ~sign2;
        do {
            if ((bits2 & sign2) == 0) return;
            unsigned long long old_bits2 = bits2;
            bits2 = atomicCAS(bit2, bits2, _sign2 & bits2);
            if (bits2 == old_bits2) {
                tgt_warp_id = idx_bit1 * 64 + idx_bit2;
                if (true && bits2 == sign2) {
                    unsigned long long sign1 = 0x1ull << idx_bit1;
                    bits1 = atomicCAS(bit1, 0x0ull, 0x0ull);
                    atomicCAS(bit1, bits1, bits1 & (~sign1));
                }
                return;
            }
            __nanosleep(1<<5);
        } while (true);
    }


    __forceinline__ __device__ void Set(
        int warp_id,
        unsigned long long* bit1, unsigned long long* bit2
        ) 
    {
        bit2 += (warp_id / 64) * 16;
        unsigned long long sign2 = 0x1ull << (warp_id % 64);
        unsigned long long bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);
        while (true) {
            unsigned long long old_bits2 = bits2;
            bits2 = atomicCAS(bit2, bits2, sign2 | bits2);
            if (old_bits2 == bits2) break;
            __nanosleep(1<<4);
        }
        if (bits2 == 0x0ull) {
            unsigned long long idx1 = warp_id / 64;
            bit1 += idx1 / 64;
            unsigned long long sign1 = 0x1ull << (idx1 % 64);
            unsigned long long bits1 = atomicCAS(bit1, 0x0ull, 0x0ull);
            atomicCAS(bit1, bits1, bits1 | sign1);
        }
    }
}

namespace Bitmap {

    // It assumes only one thread per warp executes this function 
    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Get(
        int &tgt_warp_id,
        unsigned long long* bit2,
        unsigned int num_idle_warps
        )
    {
        const unsigned int num_bits1 = ((NUM_WARPS-1)/64)+1;
        unsigned long long rnd = (clock64() * 2305843009213693951u) % 137438691328u;        
        
        int idx_bit1 = (rnd % NUM_WARPS) / 64;
        bit2 += idx_bit1;
        unsigned long long bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);
        if (bits2 == 0) return;

        unsigned int num_bits2 = idx_bit1 == (num_bits1 - 1) ? (NUM_WARPS % 64) : 64;   
        
        unsigned int idx_bit2 = Pick(rnd, num_bits2, bits2);
        //unsigned int idx_bit2 = rnd % num_bits2;
        //unsigned int idx_bit2 = __ffsll(bits2) - 1;
        unsigned long long sign2 = 0x1ull << idx_bit2;
        unsigned long long _sign2 = ~sign2;
        
        do {
            if ((bits2 & sign2) == 0) return;
            unsigned long long old_bits2 = bits2;
            bits2 = atomicCAS(bit2, bits2, _sign2 & bits2);
            if (bits2 == old_bits2) {
                tgt_warp_id = idx_bit1 * 64 + idx_bit2;
                return;
            }
            __nanosleep(1 << 5);
        } while (true);
    }


    __forceinline__ __device__ void Set(
        int warp_id,
        unsigned long long* bit2) 
    {
        bit2 += (warp_id / 64);
        unsigned long long sign2 = 0x1ull << (warp_id % 64);
        unsigned long long bits2 = atomicCAS(bit2, 0x0000000000000000, 0x0000000000000000);
        while (true) {
            unsigned long long old_bits2 = bits2;
            bits2 = atomicCAS(bit2, bits2, sign2 | bits2);
            if (old_bits2 == bits2) break;
            __nanosleep(1<<4);
        }
    }
}


namespace Stack {

    struct IdStack {
        int lock;
        unsigned long long start;
        volatile int ids[1];
    };

    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Get(int &idle_warp_id, 
        WarpsStatus* warp_status, unsigned int &num_idle_warps,
        IdStack* id_stack) 
    {
        if (0 == atomicCAS(&id_stack->lock, 0, 1)) {
            // Success to acquire a lock
            unsigned int num_warps = 0;
            warp_status->fetchWarpNum(num_idle_warps, num_warps);
            int slot = atomicCAS(&id_stack->start, 0, 0);
            if (slot > 0) {
                unsigned long long sub = 1;
                int slot = atomicAdd(&id_stack->start, (~sub) + 1) - 1; // % NUM_WARPS;
                idle_warp_id = id_stack->ids[slot];
                num_idle_warps = warp_status->subIdleWarpNum() - 1;
            } 
            atomicCAS(&id_stack->lock, 1, 0);
        }
    }

    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Set(int warp_id,
        WarpsStatus* warp_status, unsigned int &num_idle_warps,
        IdStack* id_stack)
    {
        //printf("Try to get lock, warp_id: %d num_idle_warps: %d\n", warp_id, num_idle_warps);
        unsigned int num_warps = 0;
        while (0 != atomicCAS(&id_stack->lock, 0, 1)) {
            __nanosleep(1 << 6); 
        }
        //assert(false);
        num_idle_warps = warp_status->addIdleWarpNum() + 1;
        //printf("Succeess to get lock, warp_id: %d num_idle_warps: %d\n", warp_id, num_idle_warps);
        if (num_idle_warps < NUM_WARPS) {
            int slot = atomicAdd(&id_stack->start, 1);
            id_stack->ids[slot] = warp_id;
            __threadfence();
        }
        atomicCAS(&id_stack->lock, 1, 0);   
    }
}

}}


#endif