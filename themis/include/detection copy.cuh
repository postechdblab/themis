#ifndef __TWOLVLBITMAPS_CUH__
#define __TWOLVLBITMAPS_CUH__

#include <stdio.h>
#include <assert.h>

namespace Themis { namespace Detection {

namespace TwoLvlBitmaps {

    // It assumes only one thread per warp executes this function 
    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Get(
        int &tgt_warp_id,
        unsigned long long* bit1, unsigned long long* bit2,
        unsigned int num_idle_warps
        )
    {   
        tgt_warp_id = -1;
        const unsigned int num_bits1 = ((NUM_WARPS-1)/63)+1;
        unsigned int idx_bit1 = 0xFFFFFFFF;
        unsigned long long bits1;
        unsigned long long rnd = (clock64() *  2305843009213693951u) % 137438691328u;
        if (num_idle_warps > 256) {
            idx_bit1 = rnd % num_bits1;
        } else {
            bits1 = atomicCAS(bit1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
            if (bits1) {
                // Randomly choose the one of one bits
                unsigned int idx1 = rnd % num_bits1;
                // 0 00 0001   0 0001 00
                unsigned long long left_bits = idx1 == 0 ? bits1 : bits1 << idx1;
                unsigned long long right_bits = idx1 == 0 ? 0x0ull : bits1 >> (64 - idx1);
                unsigned long long shifted_bits1 = left_bits | right_bits;
                idx_bit1 = __ffsll(shifted_bits1) - 1;
                //printf("idx1: %d, bits1: %llx, shifted_bits1: %llx, left_bits: %llx, right_bits: %llx, idx_bit1: %d\n", 
                //    idx1, bits1, shifted_bits1, left_bits, right_bits, idx_bit1);
                idx_bit1 += idx_bit1 < idx1 ? (num_bits1 - idx1 + 1) : -idx1;
                //assert(idx_bit1 < num_bits1);
                //printf("idx_bit1: %d\n", idx_bit1);
            } 
        }
        
        if (idx_bit1 != 0xFFFFFFFF) {
            //printf("Try to find in bitmap2\n");
            //unsigned long long sign1 = 0x1ull << idx_bit1;
            bit2 += idx_bit1;
            
            //while (true) {
                unsigned long long bits2 = atomicCAS(bit2, 0x0ull, 0x0ull);

                if (bits2 & 0x8000000000000000) return;
                if ((bits2 & 0x7FFFFFFFFFFFFFFF) == 0) return;

                unsigned int num_bits2 = idx_bit1 == (num_bits1 - 1) ? (NUM_WARPS % 63) : 63;
                unsigned int idx2 = rnd % 524287 % num_bits2;
                
                unsigned long long shifted_bits2 = 0x7FFFFFFFFFFFFFFFull & bits2;

                unsigned long long left_bits = idx2 == 0 ? shifted_bits2 : shifted_bits2 << idx2;
                unsigned long long right_bits = idx2 == 0 ? 0x0ull : shifted_bits2 >> (64 - idx2);

                shifted_bits2 = left_bits | right_bits;
                unsigned int idx_bit2 = __ffsll(shifted_bits2) - 1;

                idx_bit2 += idx_bit2 < idx2 ? (num_bits2 - idx2 + 1) : -idx2;

                unsigned long long sign2 = 0x1ull << idx_bit2;            
                unsigned long long old_bits2;
                if (bits2 == sign2) {
                    bits2 = atomicCAS(bit2, sign2, 0x8000000000000000);
                    if (bits2 == sign2) {
                        // Update bit1
                        unsigned long long sign1 = 0x1ull << idx_bit1;
                        unsigned long long old_bits;
                        do {
                            old_bits = bits1;
                            bits1 = atomicCAS(bit1, bits1, bits1 & (~sign1));
                            if (old_bits == bits1) break;
                            __nanosleep(1<<3);
                            
                        } while (true);
                        atomicCAS(bit2, 0x8000000000000000, 0x0ull);
                        tgt_warp_id = 63 * idx_bit1 + idx_bit2;
                        return;
                    }
                } else {
                    old_bits2 = bits2;
                    bits2 = atomicCAS(bit2, bits2, bits2 & (~sign2));
                    if (old_bits2 == bits2) {
                        tgt_warp_id = 63 * idx_bit1 + idx_bit2;
                        return;
                    }
                }
                //__nanosleep(1<<4);
            //}
        }
    }


    __forceinline__ __device__ void Set(
        int warp_id,
        unsigned long long* bit1, unsigned long long* bit2
        ) 
    {
        bit2 += warp_id / 63;
        unsigned long long sign2 = 0x1ull << (warp_id % 63);
        unsigned long long sign1 = 0x1ull << (warp_id / 63);
        //printf("Try to set ..\n");
        /*
        while (true) {
            unsigned long long bits2 = atomicCAS(bit2, 0x0000000000000000, sign2 | 0x8000000000000000);
            //assert((bits2 & sign2) == 0);
            if ((bits2 & 0x8000000000000000) == 0) {
                if (bits2 == 0) { // need to update bit1
                    unsigned long long bits1 = 0;
                    unsigned long long old_bits;
                    do {
                        old_bits = bits1;
                        bits1 = atomicCAS(bit1, bits1, sign1 | bits1);
                        if (old_bits == bits1) break;
                        __nanosleep(1<<4);
                    } while (true);
                    atomicCAS(bit2, sign2 | 0x8000000000000000, sign2 );
                    break;
                } else { // need to try to update bit2
                    unsigned long long old_bits2 = bits2;
                    bits2 = atomicCAS(bit2, bits2, sign2 | bits2);
                    if (old_bits2 == bits2) break;        
                }
            }
            __nanosleep(1<<5);
        }
        */
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
        unsigned long long rnd = (clock64() *  2305843009213693951u) % 137438691328u;
        int idx = rnd % NUM_WARPS;
        bit2 += idx / 64;
        unsigned long long sign2 = (0x1ull << (idx % 64));
        unsigned long long _sign2 = ~sign2;

        unsigned long long bits2 = atomicCAS(bit2, 0x0000000000000000, 0x0000000000000000);
        if ((bits2 & sign2) == 0) return;
        while (true) {
            unsigned long long old_bits2 = bits2;
            bits2 = atomicCAS(bit2, bits2, _sign2 & bits2);
            if (old_bits2 == bits2) break;
            if ((bits2 & sign2) == 0) return;
            __nanosleep(1<<4);
        }
        tgt_warp_id = idx;
    }


    __forceinline__ __device__ void Set(
        int warp_id,
        unsigned long long* bit2) 
    {
        bit2 += warp_id / 64;
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
        unsigned int* global_num_idle_warps, unsigned int &num_idle_warps,
        IdStack* id_stack) 
    {
        if (0 == atomicCAS(&id_stack->lock, 0, 1)) {
            // Success to acquire a lock
            num_idle_warps = atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF);
            if (num_idle_warps > 0) {
                int slot = atomicAdd(&id_stack->start, 1) % NUM_WARPS;
                idle_warp_id = id_stack->ids[slot];
                num_idle_warps = atomicSub(global_num_idle_warps, 1) - 1;                     
            } 
            atomicCAS(&id_stack->lock, 1, 0);
        }
    }

    template<unsigned long long NUM_WARPS>
    __forceinline__ __device__ void Set(int warp_id,
        unsigned int* global_num_idle_warps, unsigned int &num_idle_warps,
        IdStack* id_stack)
    {
        while (0 != atomicCAS(&id_stack->lock, 0, 1)) { 
            num_idle_warps = atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF);
            if ((num_idle_warps+1) == NUM_WARPS) break;
            __nanosleep(1 << 3); 
        }
        //assert(false);
        num_idle_warps = atomicAdd(global_num_idle_warps, 1) + 1;
        //printf("warp_id: %d num_idle_warps: %d\n", warp_id, num_idle_warps);
        if (num_idle_warps < NUM_WARPS) {
            int slot = atomicCAS(&id_stack->start, 0, 0);
            slot = (slot + num_idle_warps - 1) % NUM_WARPS;
            id_stack->ids[slot] = warp_id;
            __threadfence();
        }
        atomicCAS(&id_stack->lock, 1, 0);   
    }
}

}}


#endif