#ifndef __THEMMIS_CUH__
#define __THEMMIS_CUH__

#include "range.cuh"
#include "relation.cuh"
#include "detection.cuh"
#include "pushedparts.cuh"
#include "warp_status.cuh"

#include <iostream>

#define TS_WIDTH 64

#define STEP_SIZE_AT_ZERO (32 * 64)

namespace Themis {

    namespace Trie {

        __global__ void BuildLevel(int* trie_pre_offsets, 
            int* trie_keys, int* offsets, 
            int* flags, int* indirect, unsigned long long* keys, int n) 
        {
            int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int i = thread_id; i < n; i += step) {
                if (i == 0 || (flags[i] != flags[i-1])) {
                    unsigned long long key = keys[i];
                    trie_keys[flags[i]] = key & 0xFFFFFFFF;;
                    offsets[flags[i]] = i;
                    int prefix_k = key >> 32;
                    if (i == 0 || prefix_k != (keys[i-1] >> 32)) trie_pre_offsets[prefix_k] = flags[i];
                }
                if (i == n-1) {
                    int j = keys[i]>>32;
                    trie_pre_offsets[j+1] = flags[i]+1;
                }
            }
        }

        __global__ void SetKey(unsigned long long* keys, int* flags, int* indirect, int* vals, int n) {
            int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int i = thread_id; i < n; i += step) {
                unsigned long long new_key = flags[i];
                unsigned long long val = vals[indirect[i]];
                new_key = (new_key << 32) | val;
                keys[i] = new_key;
            }
        }

        __global__ void SetFlag(unsigned long long* keys, int* flags, int n) {
            int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int i = thread_id; i < n; i += step) {
                flags[i] = i > 0 && keys[i] != keys[i-1] ? 1 : 0;
            }
        }
    }

    int clz(int x) {
        if (x == 0) return 32;
        return __builtin_clz(x);
    }

    __device__ __forceinline__ unsigned int nth_bit_set(uint32_t value, unsigned int n)
    {
        uint32_t      mask = 0x0000FFFFu;
        unsigned int  size = 16u;
        unsigned int  base = 0u;
        if (++n > __popc(value)) return 32;
        while (size > 0) {
            const unsigned int  count = __popc(value & mask);
            if (n > count) { // go to right
                base += size;
                size >>= 1;
                mask |= mask << size;
            } else {
                size >>= 1;
                mask >>= size;
            }
        }
        return base;
    }

    __device__ int PrefixSumUp(int thread_id, int prefix_sum) {
        int buf = __shfl_up_sync(ALL_LANES, prefix_sum, 16);
        prefix_sum += thread_id >= 16 ? buf : 0;
        buf = __shfl_up_sync(ALL_LANES, prefix_sum, 8);
        prefix_sum += thread_id >= 8 ? buf : 0;
        buf = __shfl_up_sync(ALL_LANES, prefix_sum, 4);
        prefix_sum += thread_id >= 4 ? buf : 0;
        buf = __shfl_up_sync(ALL_LANES, prefix_sum, 2);
        prefix_sum += thread_id >= 2 ? buf : 0;
        buf = __shfl_up_sync(ALL_LANES, prefix_sum, 1);
        prefix_sum += thread_id >= 1 ? buf : 0;
        return prefix_sum;
    }

    __device__ __forceinline__ void PrefixSumDown(int thread_id, int &prefix_sum) {
        int buf = __shfl_down_sync(ALL_LANES, prefix_sum, 16);
        prefix_sum += thread_id < 16 ? buf : 0;
        buf = __shfl_down_sync(ALL_LANES, prefix_sum, 8);
        prefix_sum += thread_id < 24 ? buf : 0;
        buf = __shfl_down_sync(ALL_LANES, prefix_sum, 4);
        prefix_sum += thread_id < 28 ? buf : 0;
        buf = __shfl_down_sync(ALL_LANES, prefix_sum, 2);
        prefix_sum += thread_id < 30 ? buf : 0;
        buf = __shfl_down_sync(ALL_LANES, prefix_sum, 1);
        prefix_sum += thread_id < 31 ? buf : 0;
    }

    union RangeWrapper {
        unsigned long long data;
        Range range;    
    };

    __device__ __forceinline__ void PrintLTS(const char* pre, int thread_id, int ts_sid_cached, Range &ts_range_cached, Range* ts_ranges)
    {
        printf("%s, warp_id: %d, thread_id: %d, ts_sid_cached: %d, ts_range_cached: (%d,%d), ts_ranges: (%d,%d), (%d,%d)\n", 
            pre, (threadIdx.x + blockDim.x * blockIdx.x)/32, thread_id, ts_sid_cached, 
            ts_range_cached.start, ts_range_cached.end, 
            ts_ranges[0].start, ts_ranges[0].end, 
            ts_ranges[1].start, ts_ranges[1].end);        
    }
    
    /*
    __device__ __forceinline__ void FillIPartAtZeroLvl(
        int lvl, int thread_id, int &inodes_cnts, int &sid, int&tid,
        Range &ts_range_cached
    ) {
        if (thread_id == 0) inodes_cnts -= 32;
        sid = ts_range_cached.start < ts_range_cached.end ? 0 : -1;
        tid = ts_range_cached.start;
        ts_range_cached.start += 32; //blockDim.x * gridDim.x;
        ts_range_cached.start = ts_range_cached.start < ts_range_cached.end ? ts_range_cached.start : ts_range_cached.end;        
    }
    */

    __device__ __forceinline__ void UpdateMaskAtIfLvlAfterPush(int lvl, int cnt, unsigned &mask_32, unsigned &mask_1) {
        mask_32 = cnt >= 32 ? (mask_32 | (0x1 << lvl)) : (mask_32 & (~(0x1u << lvl)));    
        mask_1 =  mask_1 | (0x1 << lvl);
    };
    

    __device__ __forceinline__ void UpdateMaskAtZeroLvl(int lvl, int thread_id, Range &range, unsigned &mask_32, unsigned &mask_1) {
        int cnt = range.end - (range.start - thread_id);
        mask_32 = cnt >= 32 ? (mask_32 | (0x1 << lvl)) : (mask_32 & (~(0x1u << lvl)));    
        mask_1 = cnt > 0 ? (mask_1 | (0x1 << lvl)) : (mask_1 & (~(0x1u << lvl)));    
    }

    __device__ __forceinline__ void FillIPartAtZeroLvl(
        int lvl, int thread_id, int &active, int&tid, Range &ts_range_cached, unsigned &mask_32, unsigned &mask_1, int step
    ) {
        active = ts_range_cached.start < ts_range_cached.end;
        tid = ts_range_cached.start;
        ts_range_cached.start += step; //blockDim.x * gridDim.x;
        UpdateMaskAtZeroLvl(lvl, thread_id, ts_range_cached, mask_32, mask_1);
    }

    __device__ __forceinline__ void FillIPartAtZeroLvlForMorsel(
        int lvl, int thread_id, int &inodes_cnts, int &sid, int&tid,
        Range &ts_range_cached
    ) {
        if (thread_id == 0) inodes_cnts = inodes_cnts > 32 ? inodes_cnts - 32 : 0;
        sid = ts_range_cached.start < ts_range_cached.end ? 0 : -1;
        tid = ts_range_cached.start;
        ts_range_cached.start += 32;
        ts_range_cached.start = ts_range_cached.start < ts_range_cached.end ? ts_range_cached.start : ts_range_cached.end;
    }
    
    __device__ __forceinline__ void UpdateMaskAtIfLvl(int lvl, int cnt, unsigned &mask_32, unsigned &mask_1) {
        cnt = __shfl_sync(ALL_LANES, cnt, lvl);
        mask_32 = mask_32 & (~(0x1u << lvl));    
        mask_1 = cnt > 0 ? (mask_1 | (0x1 << lvl)) : (mask_1 & (~(0x1u << lvl)));  
    };
    

    __device__ __forceinline__ void FillIPartAtIfLvl(
        int lvl, int thread_id, int &inodes_cnts, int &active, unsigned &mask_32, unsigned &mask_1
    ) {
        int cnt = __shfl_sync(ALL_LANES, inodes_cnts, lvl);
        mask_32 = mask_32 & (~(0x1u << lvl));    
        mask_1 = cnt > 32 ? (mask_1 | (0x1 << lvl)) : (mask_1 & (~(0x1u << lvl)));  
        active = thread_id < cnt;
        inodes_cnts -= (thread_id == lvl) ? ((inodes_cnts >= 32) ? 32 : inodes_cnts) : 0;
    }

    __device__ __forceinline__ void FillIPartAtIfLvlTest(
        int lvl, int thread_id, int &cnt, int &active, unsigned &mask_32, unsigned &mask_1
    ) {
        mask_32 = mask_32 & (~(0x1u << lvl));    
        mask_1 = cnt > 32 ? (mask_1 | (0x1 << lvl)) : (mask_1 & (~(0x1u << lvl)));  
        active = thread_id < cnt;
        cnt -= 32;
    }
    
    __device__ __forceinline__ void UpdateMaskAtLoopLvl(int lvl, Range &range, unsigned &mask_32, unsigned &mask_1) {
        
        unsigned m = __ballot_sync(ALL_LANES, range.start < range.end);
        mask_32 = m == 0xFFFFFFFFu ? (mask_32 | (0x1 << lvl)) : (mask_32 & (~(0x1u << lvl)));    
        mask_1 = m != 0 ? (mask_1 | (0x1 << lvl)) : (mask_1 & (~(0x1u << lvl)));    
    }

    
    __device__ __forceinline__ void FillIPartAtLoopLvl(
        int lvl, int thread_id, int &active, int &tid, Range &dpart_range, unsigned &mask_32, unsigned &mask_1
    ) {
        active = dpart_range.start < dpart_range.end;
        tid = dpart_range.start;
        dpart_range.start += 32;
    }

    __device__ __forceinline__ void GetRangeFromTS(
        Range &range, Range* ts_ranges, int ts_sid_cached)
    {
        int ts_src = ts_sid_cached % 32;
        int ts_range_start0 = __shfl_sync(ALL_LANES, ts_ranges[0].start, ts_src);
        int ts_range_end0 = __shfl_sync(ALL_LANES, ts_ranges[0].end, ts_src);
        int ts_range_start1 = __shfl_sync(ALL_LANES, ts_ranges[1].start, ts_src);
        int ts_range_end1 = __shfl_sync(ALL_LANES, ts_ranges[1].end, ts_src);
        range.start = ts_sid_cached < 32 ? ts_range_start0 : ts_range_start1;
        range.end = ts_sid_cached < 32 ? ts_range_end0 : ts_range_end1;
    }

    __device__ __forceinline__ unsigned PushTrapezoids(
        unsigned pre_lanes, unsigned active_mask, int& sid, int lvl, int active, int thread_id, int &inodes_cnts
    ) {
        int num_active = __popc(active_mask);
        int cnt = __shfl_sync(ALL_LANES, inodes_cnts, lvl) + num_active;
        sid = active ? (TS_WIDTH - cnt) + __popc(pre_lanes & active_mask) : -1;
        if (thread_id == lvl) inodes_cnts = cnt;
        return TS_WIDTH - cnt;
    }

    __device__ __forceinline__ void PushToPart(
        int &sid, int lvl, int active, int thread_id, int &inodes_cnts,
        Range& part_range, Range &local_range
    ) {
        // 1. Count the number of new Known nodes
        int local_cnt = active ? (local_range.end - local_range.start) : 0;
        PrefixSumDown(thread_id, local_cnt);
        local_cnt = __shfl_sync(ALL_LANES, local_cnt, 0);
        // 2. Update inodes_cnts
        if (thread_id == lvl) inodes_cnts += local_cnt;
        sid = active ? thread_id : -1;
        part_range.start = active ? local_range.start : 0;
        part_range.end = active ? local_range.end : 0;
    }


    __device__ __forceinline__ void PushToPart(
        int lvl, int active, int thread_id,
        Range& part_range, Range &local_range
    ) {
        //sid = active ? thread_id : -1;
        part_range.start = active ? local_range.start : 0;
        part_range.end = active ? local_range.end : 0;
    }

    __device__ __forceinline__ bool DistributeFromPartToDPart(
        int thread_id, int lvl, int &sid_result, Range& part_range, Range& dpart_range
    ) {
        unsigned part_mask = __ballot_sync(ALL_LANES, part_range.start < part_range.end);
        unsigned dpart_mask = __ballot_sync(ALL_LANES, dpart_range.start < dpart_range.end);

        int sid = __ffs(part_mask)-1;
        int num_threads_filled = __popc(dpart_mask);
        int remaining = thread_id - num_threads_filled;

        Range range;
        do {
            range.start = __shfl_sync(ALL_LANES, part_range.start, sid);
            range.end = __shfl_sync(ALL_LANES, part_range.end, sid);
            int margin = range.size();
            if (remaining >= 0 && remaining < margin) {
                dpart_range.start = range.start + remaining;
                dpart_range.end = range.end;
                sid_result = sid;
            }
            remaining -= margin; 
        } while (__any_sync(ALL_LANES, remaining >= 0) && ++sid < 32);

        if (thread_id == 31 && sid < 32) {
            if ((range.start + 31) > dpart_range.start) dpart_range.end = dpart_range.start + 1;
        }

        if (sid < 32) {
            int range_end = __shfl_sync(ALL_LANES, dpart_range.end, 31);
            if (sid_result == sid) dpart_range.end = range_end;
            if (thread_id == sid) part_range.start = range_end;   
        }        
        if (thread_id < sid) part_range.start = part_range.end = 0;
        return true;
    }

    __device__ __forceinline__ bool DistributeFromPartToDPart(
        int thread_id, int lvl, int &sid_result, Range& part_range, Range& dpart_range, unsigned &mask_32, unsigned &mask_1
    ) {
        unsigned dpart_mask = __ballot_sync(ALL_LANES, dpart_range.start < dpart_range.end);
        if (dpart_mask == 0xFFFFFFFFu) {
            mask_32 = mask_32 | (0x1u << lvl);
            return false;
        }
        unsigned part_mask = __ballot_sync(ALL_LANES, part_range.start < part_range.end);
        if (part_mask == 0) {
            mask_32 = mask_32 & (~(0x1 << lvl));
            mask_1 = dpart_mask == 0 ? mask_1 & (~(0x1 << lvl)) : mask_1;
            return false;
        }

        int sid = __ffs(part_mask)-1;
        int num_threads_filled = __popc(dpart_mask);
        int remaining = thread_id - num_threads_filled;

        Range range;
        do {
            range.start = __shfl_sync(ALL_LANES, part_range.start, sid);
            range.end = __shfl_sync(ALL_LANES, part_range.end, sid);
            int margin = range.size();
            if (remaining >= 0 && remaining < margin) {
                dpart_range.start = range.start + remaining;
                dpart_range.end = range.end;
                sid_result = sid;
            }
            remaining -= margin; 
        } while (__any_sync(ALL_LANES, remaining >= 0) && ++sid < 32);

        if (thread_id == 31 && sid < 32) {
            if ((range.start + 31) > dpart_range.start) dpart_range.end = dpart_range.start + 1;
        }

        if (sid < 32) {
            int range_end = __shfl_sync(ALL_LANES, dpart_range.end, 31);
            if (sid_result == sid) dpart_range.end = range_end;
            if (thread_id == sid) part_range.start = range_end;
            mask_32 = mask_32 | (0x1 << lvl);    
        }        
        if (thread_id < sid) part_range.start = part_range.end = 0;
        return true;
    }

    struct GtsRange {
        volatile int start = 0;
        volatile int end = 0;
        
        __device__ __host__ GtsRange() {
            start = end  = 0;
        }
        __device__ __host__ __forceinline__ void set(int s, int e) {
            start = s;
            end = e;
        }

        __device__ __host__ __forceinline__ int size() {
            return end - start;
        }

        __device__ __host__ __forceinline__ void get(int &s, int &e) {
            s = start;
            e = end;
        }

        __device__ __host__ __forceinline__ void set(const GtsRange &copy) {
            start = copy.start;
            end = copy.end;
        }

        __device__ __host__ __forceinline__ void set(const Range &copy) {
            start = copy.start;
            end = copy.end;
        }

        __device__ __host__ __forceinline__ unsigned long long wrap() {
            unsigned long long result = ((unsigned long long) start) << 32;
            //result = result << 32;
            result |= ((unsigned long long) end) & 0x00000000FFFFFFFF;
            return result;
        }
    };

    __device__ __forceinline__ int GetLtsCnt(int thread_id, int base_sid, Range* lts_ranges) {
        int size0 = base_sid <= thread_id ? lts_ranges[0].size() : 0;
        int prefix_sum0 = PrefixSumUp(thread_id, size0);
        int size1 = base_sid <= (thread_id+32) ? lts_ranges[1].size() : 0;
        int prefix_sum1 = PrefixSumUp(thread_id, size1);
        return __shfl_sync(ALL_LANES, prefix_sum0 + prefix_sum1, 31);
    }

    __device__ __forceinline__ int GetNumIdleWarps(int* global_num_idle_warps, int thread_id) 
    {
        int num_idle_warps;
        if (thread_id ==0) num_idle_warps = atomicCAS(global_num_idle_warps, -1, -1);
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        return num_idle_warps;
    }

    
    __device__ __forceinline__ void CountINodesAtZeroLvl(int thread_id, Range &range_cached, int &num_nodes) {
        if (thread_id == 0) {
            int step = blockDim.x * gridDim.x;
            int num_all_nodes = range_cached.start < range_cached.end ? (range_cached.end - range_cached.start) : 0;
            num_nodes = num_all_nodes % step;
            num_nodes = num_nodes >= 32 ? 32 : num_nodes % 32;
            num_nodes += (num_all_nodes / step) * 32;
        }
    }

    __device__ __forceinline__ void CountINodesAtLoopLvl(int thread_id, int lvl, Range &range_cached, Range &range, int &num_nodes) {
        int cnt = range.start < range.end ? range.end - range.start : 0;
        cnt += range_cached.start < range_cached.end ? ((range_cached.end - range_cached.start - 1) >> 5) + 1 : 0;
        PrefixSumDown(thread_id, cnt);
        cnt = __shfl_sync(ALL_LANES, cnt, 0);
        if (thread_id == lvl) num_nodes = cnt;
    }

    __device__ __forceinline__ void CountINodesAtIfLvl(int thread_id, int lvl, int &num_nodes) {}

    __device__ __forceinline__ void PullINodesAtZeroLvlRangeStatically(int thread_id, Range& range_cached, int &num_nodes) {

        int num_all_nodes = range_cached.end - range_cached.start;
        int num_blocks = (num_all_nodes + 31) / 32;

        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int num_warps = blockDim.x * gridDim.x / 32;

        int num_block_per_warp = num_blocks / num_warps;
        int num_remaining_blocks = num_blocks % num_warps;

        int start = num_block_per_warp * 32 * warp_id;
        int end = start + num_block_per_warp * 32;
        if (num_remaining_blocks > 0) {
            int pivot = num_remaining_blocks - 1;
            // 1 1 1 0 0 0
            // 0 1 2 3 4 5
            if (warp_id > pivot) {
                start += 32 * num_remaining_blocks;
                end += 32 * num_remaining_blocks;
            } else {
                start += 32 * warp_id;
                end += 32 * warp_id + 32;
            }
        }
        range_cached.start = start + thread_id;
        range_cached.end = end < range_cached.end ? end : range_cached.end;
        if (thread_id == 0) num_nodes = range_cached.end - range_cached.start;
    }

    __device__ __forceinline__ void PullINodesAtZeroLvlStatically(int thread_id, Range& range_cached, int &num_nodes) {   
        int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        if (thread_id == 0) range_cached.start = 32 * warp_id;
        range_cached.start = __shfl_sync(ALL_LANES, range_cached.start, 0) + thread_id;
        CountINodesAtZeroLvl(thread_id, range_cached, num_nodes);
        return;
    }

    template<int NUM_WARPS>
    __device__ __forceinline__ bool PullINodesAtZeroLvlDynamically(int thread_id, int* global_scan_offset, int global_scan_end, int &local_scan_offset, Range& range_cached, int &num_nodes) {
        num_nodes = 0;
        range_cached.start = range_cached.end + thread_id;
        if (local_scan_offset >= global_scan_end) return false;
        if (thread_id == 0) local_scan_offset = atomicAdd(global_scan_offset, 1);
        local_scan_offset = __shfl_sync(ALL_LANES, local_scan_offset, 0);
        if ((local_scan_offset >= NUM_WARPS) || (local_scan_offset * 32 >= range_cached.end)) return false; 
        range_cached.start = local_scan_offset * 32 + thread_id;
        CountINodesAtZeroLvl(thread_id, range_cached, num_nodes);
        return true;
    }

    __device__ __forceinline__ void PullINodesFromPPartAtZeroLvl(int thread_id, PushedParts::PushedPartsAtZeroLvl* gts, Range& range_cached, int &num_nodes)
    {
        if (thread_id == 0) {
            range_cached.start = gts->start;
            range_cached.end = gts->end;
            num_nodes = gts->cnt;
        }
        range_cached.start = __shfl_sync(ALL_LANES, range_cached.start, 0) + thread_id;
        range_cached.end = __shfl_sync(ALL_LANES, range_cached.end, 0);
    }

    
    __device__ __forceinline__ int PushINodesToPPartAtZeroLvl(int thread_id, PushedParts::PushedPartsAtZeroLvl* gts, Range& range_cached, int step)
    {
        int num_to_push;
        if (thread_id == 0) {
            int num_nodes = range_cached.start < range_cached.end ? range_cached.end - range_cached.start : 0; 
            int num_blocks = num_nodes / step;
            if (num_blocks > 0) {
                int num_blocks_to_push = num_blocks >= 2 ? (num_blocks / 2) : 1;
                num_to_push = num_blocks_to_push * 32; 
                int e = range_cached.start + num_blocks_to_push * step;
                gts->start = range_cached.start;
                gts->end = e;
                range_cached.start = e;
            } else {
                num_to_push = num_nodes >= 32 ? 32 : num_nodes % 32;
                gts->start = range_cached.start;
                gts->end = range_cached.end;
                range_cached.start = range_cached.end;
            }
            gts->cnt = num_to_push;
            gts->lvl = 0;
        }
        range_cached.start = __shfl_sync(ALL_LANES, range_cached.start, 0) + thread_id;
        return __shfl_sync(ALL_LANES, num_to_push, 0);
    }


    __device__ __forceinline__ void PullINodesFromPPartAtLoopLvl(int thread_id, int lvl, 
        PushedParts::PushedPartsAtLoopLvl* gts, Range &dpart_range, Range& part_range, int &inodes_cnts) 
    {
        dpart_range.setFromGts(gts->start[thread_id], gts->end[thread_id]);
        part_range.setFromGts(gts->start[thread_id+32], gts->end[thread_id+32]);
        if (thread_id == lvl) inodes_cnts = gts->cnt;
    }

    template<typename T>
    __device__ __forceinline__ void PullAttributesAtLoopLvl(int thread_id, T &dpart_attrs, T &part_attrs, volatile T* gts_attrs)
    {
        dpart_attrs = gts_attrs[thread_id];
        part_attrs = gts_attrs[thread_id+32];
    }

    __device__ __forceinline__ void PullStrAttributesAtLoopLvl(int thread_id, str_t &dpart_attrs, str_t &part_attrs, volatile str_t* gts_attrs)
    {
        dpart_attrs.start = gts_attrs[thread_id].start;
        dpart_attrs.end = gts_attrs[thread_id].end;
        part_attrs.start = gts_attrs[thread_id+32].start;
        part_attrs.end = gts_attrs[thread_id+32].end;
    }


    __device__ __forceinline__ int PushINodesToPPartAtLoopLvl(int thread_id, int lvl, 
        PushedParts::PushedPartsAtLoopLvl* gts, Range &dpart_range, Range &part_range) 
    {
        int num_to_push = 0;
        {
            if (dpart_range.start < dpart_range.end) {
                int num = ((dpart_range.end - dpart_range.start - 1) >> 5) + 1;
                int num_to_keep = num / 2;
                num_to_push = num - num_to_keep;
                int new_end = dpart_range.start + num_to_push * 32;
                new_end = new_end < dpart_range.end ? new_end : dpart_range.end;
                gts->start[thread_id] = dpart_range.start;
                gts->end[thread_id] = new_end;
                dpart_range.start = new_end;

            } else {
                gts->start[thread_id] = 0;
                gts->end[thread_id] = 0;
            }
        }
        {
            if (part_range.start < part_range.end) {
                int num = (part_range.end - part_range.start);
                int num_to_keep = num / 2;
                int new_end = part_range.end - num_to_keep;
                gts->start[thread_id+32] = part_range.start;
                gts->end[thread_id+32] = new_end;
                part_range.start = new_end;
                num_to_push += num - num_to_keep;
            } else {
                gts->start[thread_id+32] = 0;
                gts->end[thread_id+32] = 0;
            }
        }
        PrefixSumDown(thread_id, num_to_push);
        num_to_push = __shfl_sync(ALL_LANES, num_to_push, 0);
        //int num_to_keep;
        //CountINodesAtLoopLvl(thread_id, lvl, dpart_range, part_range, num_to_keep);
        if (thread_id == lvl) {
            gts->cnt = num_to_push;
            gts->lvl = lvl;
            //printf("num_push: %d, num_remaining: %d\n", num_to_push, num_to_keep);
        }
        return num_to_push;
    }

    template<typename T>
    __device__ __forceinline__ void PushAttributesAtLoopLvl(int thread_id, volatile T* gts_attrs, T& dpart_attr, T& part_attr) 
    {
        gts_attrs[thread_id] = dpart_attr;
        gts_attrs[thread_id+32] = part_attr;
    }

    __device__ __forceinline__ void PushPtrIntAttributesAtLoopLvl(int thread_id, volatile int** gts_attrs, int* dpart_attr, int* part_attr) 
    {
        gts_attrs[thread_id] = dpart_attr;
        gts_attrs[thread_id+32] = part_attr;
    }

    __device__ __forceinline__ void PushStrAttributesAtLoopLvl(int thread_id, volatile str_t* gts_attrs, str_t& dpart_attr, str_t& part_attr) 
    {
        *((volatile char**)(&gts_attrs[thread_id].start)) = dpart_attr.start;
        *((volatile char**)(&gts_attrs[thread_id].end)) = dpart_attr.end;
        *((volatile char**)(&gts_attrs[thread_id+32].start)) = part_attr.start;
        *((volatile char**)(&gts_attrs[thread_id+32].end)) = part_attr.end;
    }


    __device__ __forceinline__ void PullINodesFromPPartAtIfLvl(int thread_id, int lvl, PushedParts::PushedPartsAtIfLvl* gts, int &inodes_cnts) 
    {
        if (thread_id == lvl) inodes_cnts = gts->cnt;
    }

    template<typename T>
    __device__ __forceinline__ void PullAttributesAtIfLvl(int thread_id, T &lts_attr, volatile T* gts_attrs)
    {
        lts_attr = gts_attrs[thread_id];
    }

    __device__ __forceinline__ void PullStrAttributesAtIfLvl(int thread_id, str_t &lts_attr, volatile str_t* gts_attrs)
    {
        lts_attr.start = gts_attrs[thread_id].start;
        lts_attr.end = gts_attrs[thread_id].end;
    }

    __device__ __forceinline__ int PushINodesToPPartAtIfLvl(int thread_id, int lvl, PushedParts::PushedPartsAtIfLvl* gts, int &inodes_cnts) 
    {
        int num_to_push = __shfl_sync(ALL_LANES, inodes_cnts, lvl);
        if (thread_id == lvl) {
            gts->cnt = inodes_cnts;
            gts->lvl = lvl;
            inodes_cnts = 0;
        }
        return num_to_push;
    }

    template<typename T>
    __device__ __forceinline__ void PushAttributesAtIfLvl(int thread_id, volatile T* gts_attrs, T &lts_attr) 
    {
        gts_attrs[thread_id] = lts_attr;
    }

    __device__ __forceinline__ void PushPtrIntAttributesAtIfLvl(int thread_id, volatile int** gts_attrs, int* &lts_attr) 
    {
        gts_attrs[thread_id] = lts_attr;
    }

    __device__ __forceinline__ void PushStrAttributesAtIfLvl(int thread_id, volatile str_t* gts_attrs, str_t &lts_attr) 
    {
        *((volatile char**)(&gts_attrs[thread_id].start)) = lts_attr.start;
        *((volatile char**)(&gts_attrs[thread_id].end)) = lts_attr.end;
    }


    __global__ void InitializeTwoLvlBitmap(
        unsigned long long* bit1,
        unsigned long long* bit2,
        int num_warps
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i == 0) bit1[0] = 0;
        for (; i < num_warps; i += blockDim.x * gridDim.x) {
            bit2[i] = 0;
        }
    }

    __device__ __forceinline__ int CalculateHighestLvl(int inodes_cnts) {
        unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
        // 0010 
        int num_zeros = __clz(active_mask_more_than_0);
        return 32 - num_zeros - 1;
    }

    __device__ __forceinline__ void ChooseLvl(
        int thread_id,
        unsigned mask_32, unsigned mask_1,
        int &lvl
        )
    {
        if (mask_32 != 0) {
            lvl = 32 - __clz(mask_32) - 1;
            return;
        }
        lvl = mask_1 != 0 ? __ffs(mask_1) - 1 : -1;
    }


    __device__ __forceinline__ void ChooseLvl(
        int thread_id,
        int inodes_cnts,
        int &lvl
        )
    {
        unsigned active_mask_more_than_32 = __ballot_sync(ALL_LANES, inodes_cnts >= 32);
        if (active_mask_more_than_32 != 0) {
            lvl = 32 - __clz(active_mask_more_than_32) - 1;
            return;
        }
        unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
        lvl = active_mask_more_than_0 != 0 ? __ffs(active_mask_more_than_0) - 1 : -1;
    }

    __device__ __forceinline__ void ChooseLvl0(
        int thread_id,
        int inodes_cnts,
        int &lvl
        )
    {
        int num = __shfl_sync(ALL_LANES, inodes_cnts, 0);
        lvl =  num > 0 ? 0 : -1;
    }

    __device__ __forceinline__ void ChooseLvl(int &lvl,
        int thread_id, int inodes_cnts, int num_nodes_at_loop_lvl, int loop_lvl
    ) {
        if (num_nodes_at_loop_lvl >= 32) lvl = loop_lvl;
        else Themis::ChooseLvl(thread_id, inodes_cnts, lvl);
    }
}

#endif
