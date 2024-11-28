#ifndef __THEMMIS_CUH__
#define __THEMMIS_CUH__

#include "relation.cuh"
#include "detection.cuh"
#include "pushedparts.cuh"

#define TS_WIDTH 64


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



    __device__ __forceinline__ int CalculateOrder(int cnt) {
        // if cnt is zero --> order is -1
        // if cnt < 32 --> order is 0
        // if cnt < 64 --> order is 1
        return cnt == 0 ? -1 : 32 - __clz(cnt >> 5); 
    }

    //order: 0 --> 1 <= cnt < 32
    //order: 1 --> cnt >= 32 & cnt < 64
    //order: 2 --> cnt >= 64 & cnt < 128
    __device__ __forceinline__ int CalculateLowerBoundOfOrder(int8_t order) {
        return order == 0 ? 1 : 0x10 << (order);
    }

    __device__ __forceinline__ int CalculateUpperBoundOfOrder(int8_t order) {
        return 0x20 << (order);
    }



    struct StatisticsPerLvl {
        unsigned long long cnt;
        unsigned int num_warps;
        unsigned int sub_num_warps[32];
        int max_order;
        unsigned long long num_inodes;
        unsigned long long num_ws;
        unsigned long long num_inodes_at_that_time;
        unsigned long long num_nodes;
        unsigned long long max_inodes_cnt;
    };



    __global__ void krnl_InitStatisticsPerLvl(
        StatisticsPerLvl* stats, unsigned int num_warps, int num_inodes_at_zero, int depth
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < depth) stats[i].max_order = 31;
        if (i == 0 && num_inodes_at_zero > 0) {
            int num_inodes_of_last_block = num_inodes_at_zero % 32;
            //printf("num_inodes_of_last_block: %d\n", num_inodes_of_last_block);
            if (num_inodes_at_zero % 32 == 0) num_inodes_of_last_block = 32;
            int num_blocks = (num_inodes_at_zero / 32) + (num_inodes_of_last_block > 0 ? 1 : 0);

            assert(num_blocks > 0);
            
            //printf("num_blocks: %d\n", num_blocks);
            int idx_case_1 = (num_blocks + num_warps -1) % num_warps;
            //printf("idx_case_1: %d\n", idx_case_1);
            int num_case_0 = idx_case_1;
            int num_inodes_of_case_0 = 32 * (num_blocks / num_warps) + 32;
            int8_t order = CalculateOrder(num_inodes_of_case_0);
            stats[0].max_order = order;
            stats[0].sub_num_warps[order] = num_case_0; 
            
            int num_case_1 = 1;
            int num_inodes_of_case_1 = 32 * (num_blocks / num_warps) + num_inodes_at_zero % 32;
            //printf("num_inodes_of_case_1: %d, num_inodes_of_last_block: %d\n", num_inodes_of_case_1, num_inodes_at_zero % 32);
            order = CalculateOrder(num_inodes_of_case_1);
            stats[0].sub_num_warps[order] += num_case_1; 

            int num_case_2 = num_warps - num_case_0 - 1;
            int num_inodes_of_case_2 = 32 * (num_blocks / num_warps);
            order = CalculateOrder(num_inodes_of_case_2);
            stats[0].sub_num_warps[order] += num_case_2; 

            if (num_inodes_of_case_2 > 0) {
                stats[0].num_warps = num_warps;
            } else {
                stats[0].num_warps = num_case_0 + 1;
            }
            //printf("stats[0].max_order: %d\n", stats[0].max_order);
            //printf("initial num warps in orders: %d/%d, %d/%d, %d/%d\n", )
            /*
            if ((num_case_0 * num_inodes_of_case_0 + 1 * num_inodes_of_case_1 + num_case_2 * num_inodes_of_case_2)
                 != num_inodes_at_zero) {
                    printf("num_warps: %d vs %d, num_inodes_at_zero: %d vs %d\n", num_warps, num_case_0 + num_case_1 + num_case_2, num_inodes_at_zero, num_case_0 * num_inodes_of_case_0 + 1 * num_inodes_of_case_1 + num_case_2 * num_inodes_of_case_2);
                    printf("num_inodes_of_last_block: %d\n", num_inodes_of_last_block);
                    printf("num_case_0: %d, num_inodes_of_case_0: %d\n", num_case_0, num_inodes_of_case_0);
                    printf("num_case_1: %d, num_inodes_of_case_1: %d\n", num_case_1, num_inodes_of_case_1);
                    printf("num_case_2: %d, num_inodes_of_case_2: %d\n", num_case_2, num_inodes_of_case_2);
                 }
            assert((num_case_0 * num_inodes_of_case_0 + 1 * num_inodes_of_case_1 + num_case_2 * num_inodes_of_case_2)
                 == num_inodes_at_zero);
            */
        }
        //stats[i].cnt = 0;
        //stats[i].num_warps = blockIdx.x == 0 && threadIdx.x == 0 ? num_warps : 0;
        //stats[i].num_inodes = 0;
        //stats[i].num_inodes_at_that_time = 0;
        //stats[i].num_nodes = 0;
        //stats[i].num_ws = 0;
        //stats[i].max_inodes_cnt = 0;
        //stats[i].max_range = 0;
        //for (int j = 0; j < 32; ++j) stats[i].sub_num_warps[j] = 0;
    }

    void InitStatisticsPerLvl(StatisticsPerLvl* &device_stats, unsigned int num_warps) {
        cudaMalloc((void**)&device_stats, sizeof(StatisticsPerLvl) * 128);
        cudaMemset(device_stats, 0, sizeof(StatisticsPerLvl) * 128);
        //krnl_InitStatisticsPerLvl<<<1,128>>>(device_stats, num_warps, depth);
    }

    void PrintStatisticsPerLvl(StatisticsPerLvl* device_stats, const char* krnl_name, int depth, unsigned int num_warps) {
        StatisticsPerLvl host_stats[128];
        cudaMemcpy(host_stats, device_stats, sizeof(StatisticsPerLvl) * depth, cudaMemcpyDeviceToHost);
        for (int i = 0; i < depth; ++i) {
            printf("%s %d # push operations: %llu\n", krnl_name, i, host_stats[i].num_ws);
            //printf("%s_%d stats: %d, %d, %llu, %llu, %llu, %lf, %lf\n", 
            //printf("%s lvl, max # inodes, # work stealings, # inodes, # nodes, # inodes / # nodes, 32 * # num ws / # num nodes : %d, %d, %llu, %llu, %llu, %lf, %lf\n", 
                //krnl_name, i, host_stats[i].max_inodes_cnt, host_stats[i].num_ws, host_stats[i].num_inodes_at_that_time, 
                //host_stats[i].num_nodes, (double)host_stats[i].num_inodes_at_that_time / (double)host_stats[i].num_nodes, 
                //64.0 * (double)host_stats[i].num_ws / (double)host_stats[i].num_nodes);
        }
        //cudaMemset(device_stats, 0, sizeof(StatisticsPerLvl) * depth);
        //krnl_InitStatisticsPerLvl<<<1,64>>>(device_stats, num_warps);
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


    struct Range {
        int start = 0;
        int end = 0;
        
        __device__ __host__ Range() {
            start = end  = 0;
        }
        __device__ __host__ __forceinline__ void set(int s, int e) {
            start = s;
            end = e;
        }

        __device__ __host__ __forceinline__ void setFromGts(volatile int &s, volatile int &e) {
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

        __device__ __host__ __forceinline__ void set(const Range &copy) 
        {
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
    
    __device__ __forceinline__ void FillIPartAtZeroLvl(
        int lvl, int thread_id, int &inodes_cnts, int &sid, int&tid,
        Range &ts_range_cached
    ) {
        if (thread_id == 0) inodes_cnts = inodes_cnts > 32 ? inodes_cnts - 32 : 0;
        sid = ts_range_cached.start < ts_range_cached.end ? 0 : -1;
        tid = ts_range_cached.start;
        ts_range_cached.start += blockDim.x * gridDim.x;
        ts_range_cached.start = ts_range_cached.start < ts_range_cached.end ? ts_range_cached.start : ts_range_cached.end;
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
    

    __device__ __forceinline__ void FillIPartAtIfLvl(
        int lvl, int thread_id, int &inodes_cnts, int &sid
    ) {
        int cnt = __shfl_sync(ALL_LANES, inodes_cnts, lvl);
        int hand = 64 - cnt;
        sid = (hand < 32) ? 
            (thread_id < hand ? 1 : 0) : ((32 + thread_id) < hand ? -1 : 1);
        
        inodes_cnts -= (thread_id == lvl) ? ((cnt >= 32) ? 32 : cnt) : 0;
        //if (thread_id == lvl) inodes_cnts -= 
    }

    __device__ __forceinline__ void FlushToIPartAtIfLvl(
        int lvl, int thread_id, unsigned push_active_mask, int &inodes_cnts, int* acive_thread_ids,  int &, unsigned &ts_src, int &ts_idx
    ) {

        
    }

    
    __device__ __forceinline__ void FillIPartAtLoopLvl(
        int lvl, int thread_id, int &inodes_cnts, int &sid, int &tid, Range &dpart_range
    ) {
        inodes_cnts -=  thread_id == lvl ? (inodes_cnts >= 32 ? 32 : inodes_cnts) : 0;
        if (dpart_range.start >= dpart_range.end) return;
        sid = thread_id;
        tid = dpart_range.start;
        dpart_range.start += 32;
        dpart_range.start = dpart_range.start < dpart_range.end ? dpart_range.start : dpart_range.end;
        //if (thread_id == lvl) inodes_cnts = inodes_cnts > 32 ? inodes_cnts - 32 : 0;
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


    __device__ __forceinline__ bool DistributeFromPartToDPart(
        int thread_id, int &sid_result, Range& part_range, Range& dpart_range
    ) {
        unsigned dpart_mask = __ballot_sync(ALL_LANES, dpart_range.end > dpart_range.start);
        if (dpart_mask == 0xFFFFFFFF) return false;
        unsigned part_mask = __ballot_sync(ALL_LANES, part_range.end > part_range.start);
        if (part_mask == 0) return false;
        
        int num_threads_filled = __popc(dpart_mask);
        int remaining = thread_id - num_threads_filled;
        int sid = __ffs(part_mask)-1;
        /*
        int local_cnt1 = part_range.end - part_range.start;
        local_cnt1 +=  dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
        PrefixSumDown(thread_id, local_cnt1);
        local_cnt1 = __shfl_sync(ALL_LANES, local_cnt1, 0);
        */

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
        //__syncwarp();
        /*
        int local_cnt2 = part_range.end - part_range.start;
        local_cnt2 +=  dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
        PrefixSumDown(thread_id, local_cnt2);
        local_cnt2 = __shfl_sync(ALL_LANES, local_cnt2, 0);
        

        
        if (blockIdx.x == 0 && threadIdx.x < 32) {
            
            if (local_cnt1 != local_cnt2) {
                printf("p0\t%d\t%d\t%d\n", part_range_0.start, part_range_0.end, part_range_0.end-part_range_0.start);
                printf("d0\t%d\t%d\t%d\n", dpart_range_0.start, dpart_range_0.end,(dpart_range_0.end-dpart_range_0.start-1)/32 + 1);
                printf("p\t%d\t%d\t%d\n", part_range.start, part_range.end, part_range.end-part_range.start);
                printf("d\t%d\t%d\t%d\n", dpart_range.start, dpart_range.end, (dpart_range.end-dpart_range.start-1)/32+1);

            }
            if (thread_id == 0 && local_cnt1 != local_cnt2) {
                printf("local_cnt1: %d, local_cnt2: %d\n", local_cnt1, local_cnt2);
            }
            //assert(local_cnt1 == local_cnt2);
        }
        */
       
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


    __device__ __forceinline__ int PullINodesFromPPartAtIfLvl(int busy_warp_id, int thread_id, int lvl,
        PushedParts::PushedPartsAtIfLvl* gts, int &inodes_cnts) 
    {
        if (thread_id == lvl) {
            inodes_cnts = gts->cnt;
        }
        return __shfl_sync(ALL_LANES, inodes_cnts, lvl);
    }

    __device__ __forceinline__ int PushINodesToPPartAtIfLvl(int idle_warp_id, int thread_id, int lvl, 
        PushedParts::PushedPartsAtIfLvl* gts, int &inodes_cnts) 
    {   
        int num_to_give;
        if (thread_id == lvl) {
            num_to_give = (inodes_cnts >= (32 * 2)) ? (inodes_cnts / 2) : (inodes_cnts >= 32 ? 32 : inodes_cnts);
            gts->cnt = num_to_give;
            gts->lvl = lvl;
            inodes_cnts -= num_to_give;
        }
        return __shfl_sync(ALL_LANES, num_to_give, lvl);
    }

    __device__ __forceinline__ void PullINodesFromPPartAtLoopLvl(int busy_warp_id, int thread_id, int lvl, 
        PushedParts::PushedPartsAtLoopLvl* gts, int &inodes_cnts, 
        Range &dpart_range, Range& part_range) 
    {
        dpart_range.setFromGts(gts->start[thread_id], gts->end[thread_id]);
        part_range.setFromGts(gts->start[thread_id+32], gts->end[thread_id+32]);

        if (thread_id == lvl) {
            inodes_cnts = gts->cnt;
        }
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

    __device__ __forceinline__ void PushINodesToPPartAtLoopLvl(int warp_id, int idle_warp_id, int thread_id, int lvl, 
        PushedParts::PushedPartsAtLoopLvl* gts, int &inodes_cnts, Range &dpart_range) 
    {
        {
            int num = dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
            int num_to_keep = num / 2;
            int new_end = dpart_range.start + (num - num_to_keep) * 32;
            new_end = new_end < dpart_range.end ? new_end : dpart_range.end;
            gts->start[thread_id] = dpart_range.start;
            gts->end[thread_id] = new_end;
            dpart_range.start = new_end;
        }
        {
            gts->start[thread_id+32] = 0;
            gts->end[thread_id+32] = 0;
        }
        int num_to_keep = dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
        PrefixSumDown(thread_id, num_to_keep);
        num_to_keep = __shfl_sync(ALL_LANES, num_to_keep, 0);
        if (thread_id == lvl) {
            //int num_to_give = inodes_cnts - num_to_keep;
            gts->cnt = inodes_cnts - num_to_keep;
            gts->lvl = lvl;
            inodes_cnts = num_to_keep;
        }
    }
    __device__ __forceinline__ void PushINodesToPPartAtLoopLvl(int warp_id, int idle_warp_id, int thread_id, int lvl, 
        PushedParts::PushedPartsAtLoopLvl* gts, int &inodes_cnts, Range &dpart_range, Range &part_range) 
    {
        {
            int num = dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
            int num_to_keep = num / 2;
            int new_end = dpart_range.start + (num - num_to_keep) * 32;
            new_end = new_end < dpart_range.end ? new_end : dpart_range.end;
            gts->start[thread_id] = dpart_range.start;
            gts->end[thread_id] = new_end;
            dpart_range.start = new_end;
        }
        {
            int num_to_keep = (part_range.end - part_range.start) / 2;
            gts->start[thread_id+32] = part_range.start;
            gts->end[thread_id+32] = part_range.end - num_to_keep;
            part_range.start = part_range.end - num_to_keep;
        }
        int num_to_keep = dpart_range.end > dpart_range.start ? (dpart_range.end - dpart_range.start-1)/32 + 1 : 0;
        num_to_keep += (part_range.end - part_range.start);
        PrefixSumDown(thread_id, num_to_keep);
        num_to_keep = __shfl_sync(ALL_LANES, num_to_keep, 0);
        if (thread_id == lvl) {
            //int num_to_give = inodes_cnts - num_to_keep;
            gts->cnt = inodes_cnts - num_to_keep;
            gts->lvl = lvl;
            inodes_cnts = num_to_keep;
        }
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


    __device__ __forceinline__ int PushINodesToPPartAtIfLvl(int dst_warp_id, int thread_id, int lvl, PushedParts::PushedPartsAtIfLvl* gts,
        int &linodes_cnts, int proportion) 
    {
        int num_to_give;
        if (thread_id == lvl) {
            num_to_give = (linodes_cnts >= (32 * proportion)) ? (linodes_cnts / proportion) : (linodes_cnts >= 32 ? 32 : linodes_cnts);
            gts->cnt = num_to_give;
            gts->lvl = lvl;
            linodes_cnts -= num_to_give;
        }
        return __shfl_sync(ALL_LANES, num_to_give, lvl);
    }

    // For the first scan operator
    __device__ __forceinline__ void PullINodesFromPPartAtZeroLvl(int busy_warp_id, int thread_id, 
        PushedParts::PushedPartsAtZeroLvl* gts, int &inodes_cnts, Range& lts_range_cached)//, int proportion) 
    {
        if (thread_id == 0) {
            lts_range_cached.start = gts->start;
            lts_range_cached.end = gts->end;
            inodes_cnts = gts->cnt;
        }
        lts_range_cached.start = __shfl_sync(ALL_LANES, lts_range_cached.start, 0) + thread_id;
        lts_range_cached.end = __shfl_sync(ALL_LANES, lts_range_cached.end, 0);
    }

    
    __device__ __forceinline__ void PushINodesToPPartAtZeroLvl(int dst_warp_id, int thread_id, 
        PushedParts::PushedPartsAtZeroLvl* gts, int &inodes_cnts, Range& lts_range_cached, int proportion)
    {
        if (thread_id == 0) {
            int num_blocks = lts_range_cached.size() / (blockDim.x * gridDim.x);
            if (num_blocks > 0) {
                int num_blocks_to_give = num_blocks >= proportion ? (num_blocks / proportion) : 1;
                int num_to_give = num_blocks_to_give * 32; 
                int e = lts_range_cached.start + num_blocks_to_give * blockDim.x * gridDim.x;
                gts->start = lts_range_cached.start;
                gts->end = e;
                lts_range_cached.start = e;
                gts->cnt = num_to_give;
                inodes_cnts -= num_to_give;
            } else {
                gts->start = lts_range_cached.start;
                gts->end = lts_range_cached.end;
                lts_range_cached.start = lts_range_cached.end;
                gts->cnt = inodes_cnts;
                inodes_cnts = 0;
            }
            gts->lvl = 0;
        }
        lts_range_cached.start = __shfl_sync(ALL_LANES, lts_range_cached.start, 0) + thread_id;
    }
    
    

    template<typename T>
    __device__ __forceinline__ void PullAttributesAtIfLvl(int thread_id, T* lts_attrs, int lts_start, volatile T* gts_attrs, int gts_start, int num)
    {
        //gts_.. hmm?? yes!!
        // thread_id - lts_start ?? 
        int idx = thread_id - lts_start;
        gts_attrs += gts_start;
        if ((idx >= 0) && (idx < num)) lts_attrs[0] = gts_attrs[idx];
        idx += 32;
        if ((idx >= 0) && (idx < num)) lts_attrs[1] = gts_attrs[idx];
    }

    __device__ __forceinline__ void PullStrAttributesAtIfLvl(int thread_id, str_t* lts_attrs, int lts_start, volatile str_t* gts_attrs, int gts_start, int num)
    {
        //gts_.. hmm?? yes!!
        // thread_id - lts_start ?? 
        int idx = thread_id - lts_start;
        gts_attrs += gts_start;
        if ((idx >= 0) && (idx < num)) {
            lts_attrs[0].start = gts_attrs[idx].start;
            lts_attrs[0].end = gts_attrs[idx].end;
        }
        idx += 32;
        if ((idx >= 0) && (idx < num)) {
            lts_attrs[1].start = gts_attrs[idx].start;
            lts_attrs[1].end = gts_attrs[idx].end;
        }
    }

    template<typename T>
    __device__ __forceinline__ void PushAttributesAtIfLvl(int thread_id, volatile T* gts_attrs, int gts_start, T* lts_attrs, int lts_start, int num) 
    {
        int idx = thread_id - lts_start;
        gts_attrs += gts_start;
        if ((idx >= 0) && (idx < num)) gts_attrs[idx] = lts_attrs[0];
        idx += 32;
        if ((idx >= 0) && (idx < num)) gts_attrs[idx] = lts_attrs[1];
    }

    __device__ __forceinline__ void PushPtrIntAttributesAtIfLvl(int thread_id, volatile int** gts_attrs, int gts_start, int** lts_attrs, int lts_start, int num) 
    {
        int idx = thread_id - lts_start;
        gts_attrs += gts_start;
        if ((idx >= 0) && (idx < num)) gts_attrs[idx] = lts_attrs[0];
        idx += 32;
        if ((idx >= 0) && (idx < num)) gts_attrs[idx] = lts_attrs[1];
    }

    __device__ __forceinline__ void PushStrAttributesAtIfLvl(int thread_id, volatile str_t* gts_attrs, int gts_start, str_t* lts_attrs, int lts_start, int num) 
    {
        int idx = thread_id - lts_start;
        gts_attrs += gts_start;
        if ((idx >= 0) && (idx < num)) {
            *((volatile char**)(&gts_attrs[idx].start)) = lts_attrs[0].start;
            *((volatile char**)(&gts_attrs[idx].end)) = lts_attrs[0].end;
        }
        idx += 32;
        if ((idx >= 0) && (idx < num)) {
            *((volatile char**)(&gts_attrs[idx].start)) = lts_attrs[1].start;
            *((volatile char**)(&gts_attrs[idx].end)) = lts_attrs[1].end;
        };
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


    
    __device__ __forceinline__ void FetchLvlAndOrder(
        int thread_id,
        int &lowest_lvl, int &globally_lowest_lvl,
        unsigned int &num_warps_which_can_push, 
        int8_t &order, int8_t &globally_current_max_order, int8_t &globally_max_order,
        int inodes_cnts,
        StatisticsPerLvl* global_stats_per_lvl,
        int depth 
        )
    {

        unsigned int num_warps_at_globally_lowest_lvl;
        if (thread_id == lowest_lvl) {
            //assert(globally_lowest_lvl <= lowest_lvl);
            // Update the globally lowest lvel
            //int old_globally_lowest_lvl = globally_lowest_lvl;
            num_warps_at_globally_lowest_lvl = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].num_warps, 0xFFFFFFFF, 0xFFFFFFFF);
            if (num_warps_at_globally_lowest_lvl == 0) { // find the new globally lowest_lvl
                while (globally_lowest_lvl < depth) {
                    num_warps_at_globally_lowest_lvl = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].num_warps, 0xFFFFFFFF, 0xFFFFFFFF);
                    if (num_warps_at_globally_lowest_lvl > 0) break;
                    globally_lowest_lvl++;
                }
                //assert(globally_lowest_lvl <= lowest_lvl);
                if (globally_lowest_lvl == lowest_lvl) {
                    atomicAdd(&global_stats_per_lvl[lowest_lvl].num_inodes, inodes_cnts);
                    num_warps_at_globally_lowest_lvl = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].num_warps, 0xFFFFFFFF, 0xFFFFFFFF);
                }
            }
            //printf("Try to update globally lowest lvl: %d\n", globally_lowest_lvl);
            //assert(lowest_lvl >= globally_lowest_lvl);
            
            // Update the maximum order if lowest_lvl is the globally_lowest_lvl
            if (lowest_lvl == globally_lowest_lvl) {
                
                globally_current_max_order = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].max_order,0xFFFFFFFF, 0xFFFFFFFF);
                //printf("Try to update globally current maximum order\n");
                num_warps_which_can_push = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].sub_num_warps[globally_current_max_order],0xFFFFFFFF, 0xFFFFFFFF);
                if (num_warps_which_can_push == 0) {
                    while (--globally_current_max_order >= 0) {
                        num_warps_which_can_push = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].sub_num_warps[globally_current_max_order],0xFFFFFFFF, 0xFFFFFFFF);
                        if (num_warps_which_can_push > 0) break;
                    }
                    atomicMin(&global_stats_per_lvl[globally_lowest_lvl].max_order, globally_current_max_order);
                }
            }
        }
        num_warps_which_can_push = __shfl_sync(ALL_LANES, num_warps_at_globally_lowest_lvl, lowest_lvl);
        globally_lowest_lvl = __shfl_sync(ALL_LANES, globally_lowest_lvl, lowest_lvl);
    }
    

    __device__ __forceinline__ int CalculateHighestLvl(int inodes_cnts) {
        unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
        // 0010 
        int num_zeros = __clz(active_mask_more_than_0);
        return 32 - num_zeros - 1;
    }

    __device__ __forceinline__ void _UpdateLvlAndOrder(
        int thread_id,
        int inodes_cnts,
        int lvl,
        int &lowest_lvl, int &globally_lowest_lvl,
        int8_t &locally_current_order, int8_t &globally_current_max_order, int8_t &globally_max_order,
        int &inodes_cnts_at_lowest_lvl,
        StatisticsPerLvl* global_stats_per_lvl
        )
    {       
        if (inodes_cnts_at_lowest_lvl == 0) {
            // We need to change the lowest level
            
            unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
            int new_lowest_lvl = active_mask_more_than_0 != 0 ? __ffs(active_mask_more_than_0) - 1 : -1;
            
            #ifdef MODE_THEMIS
            if (thread_id == new_lowest_lvl) {
                locally_current_order = CalculateOrder(inodes_cnts);
                atomicAdd(&global_stats_per_lvl[new_lowest_lvl].sub_num_warps[locally_current_order], 1);
                atomicAdd(&global_stats_per_lvl[new_lowest_lvl].num_warps, 1);
            }
            #endif

            if (new_lowest_lvl != -1) inodes_cnts_at_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, new_lowest_lvl);
            else inodes_cnts_at_lowest_lvl = 0;

            #ifdef MODE_THEMIS
            if (thread_id == lowest_lvl) {
                atomicSub(&global_stats_per_lvl[lowest_lvl].num_warps, 1);
                atomicSub(&global_stats_per_lvl[lowest_lvl].sub_num_warps[locally_current_order], 1);
                locally_current_order = -1;
            }
            #endif
            lowest_lvl = new_lowest_lvl;
        } 
        #ifdef MODE_THEMIS
        else if (thread_id == lowest_lvl && inodes_cnts < CalculateLowerBoundOfOrder(locally_current_order)) {
            // need to change current order
            //assert(inodes_cnts_at_lowest_lvl == inodes_cnts);
            int8_t new_order = CalculateOrder(inodes_cnts); // locally_current_order - 1; //CalculateOrder(inodes_cnts);
            atomicAdd(&global_stats_per_lvl[lowest_lvl].sub_num_warps[new_order], 1);
            atomicSub(&global_stats_per_lvl[lowest_lvl].sub_num_warps[locally_current_order], 1);
            locally_current_order = new_order;
        }
        #endif

        // Update level
    }

    __device__ __forceinline__ void UpdateLvlAndOrder(
        int thread_id,
        int inodes_cnts,
        int lvl,
        int &lowest_lvl, int &globally_lowest_lvl,
        int8_t &locally_current_order, int8_t &globally_current_max_order, int8_t &globally_max_order,
        int &inodes_cnt_at_lowest_lvl,
        StatisticsPerLvl* global_stats_per_lvl
        )
    {
        if (lvl != lowest_lvl) return;
        inodes_cnt_at_lowest_lvl = inodes_cnt_at_lowest_lvl > 32 ? inodes_cnt_at_lowest_lvl - 32 : 0;

        //if (thread_id == lowest_lvl) assert(inodes_cnt_at_lowest_lvl == inodes_cnts);
        
        #ifdef MODE_THEMIS
        _UpdateLvlAndOrder(thread_id, inodes_cnts, lvl, lowest_lvl, globally_lowest_lvl,
            locally_current_order, globally_current_max_order, globally_max_order,
            inodes_cnt_at_lowest_lvl,
            global_stats_per_lvl);
        #else
        if (inodes_cnt_at_lowest_lvl == 0) {
            // We need to change the lowest level
            unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
            int new_lowest_lvl = active_mask_more_than_0 != 0 ? __ffs(active_mask_more_than_0) - 1 : -1;
            lowest_lvl = new_lowest_lvl;

            if (new_lowest_lvl != -1) inodes_cnt_at_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, new_lowest_lvl);
            else inodes_cnt_at_lowest_lvl = 0;
        }
        #endif
        
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
        //unsigned active_mask_more_than_32 = __ballot_sync(ALL_LANES, inodes_cnts >= 32);
        //unsigned active_mask_more_than_0 = __ballot_sync(ALL_LANES, inodes_cnts > 0);
        //if (active_mask_more_than_32 != 0) lvl = 32 - __clz(active_mask_more_than_32) - 1;
        //else lvl = active_mask_more_than_0 != 0 ? __ffs(active_mask_more_than_0) - 1 : -1;
    }


    template<int DEPTH, uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ __forceinline__ void FindIdleWarp(
        int &idle_warp_id, int warp_id, int thread_id,
        int &lowest_lvl,  int &globally_lowest_lvl,
        int8_t &locally_current_order, int8_t &globally_current_max_order, int8_t &globally_max_order,
        unsigned int* global_num_idle_warps, unsigned int* gpart_ids, unsigned int &num_idle_warps, unsigned int &num_warps_which_can_push,
        StatisticsPerLvl* global_stats_per_lvl, int &inodes_cnts,
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp
#ifdef MODE_LB_TWOLVLBITMAPS
        , unsigned long long* bit1, unsigned long long* bit2
#endif
#ifdef MODE_LB_RANDOMIZED
        , unsigned long long* bit2
#endif
#ifdef MODE_LB_SIMPLE
        , Detection::Stack::IdStack* id_stack
#else 
#endif
        )
    {
        idle_warp_id = -1;

        
        if (thread_id == 0) num_idle_warps = atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF);
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, lowest_lvl);

        unsigned int num_busy_warps = NUM_WARPS - num_idle_warps;

        if (num_busy_warps > MIN_NUM_WARPS) {
            num_warps_which_can_push = NUM_WARPS;
            idle_warp_id = __shfl_sync(ALL_LANES, idle_warp_id, lowest_lvl);
            assert(false);
            return;
        }

        assert(NUM_WARPS == MIN_NUM_WARPS);

        num_idle_warps = MIN_NUM_WARPS - num_busy_warps;

        
        
        
        // Strategies 6 - Get the locally and globally lowest levels
        // Update the number of inodes at the globllay lowest level
    #ifdef MODE_THEMIS
        FetchLvlAndOrder(thread_id, lowest_lvl, globally_lowest_lvl, num_warps_which_can_push, 
            locally_current_order, globally_current_max_order, globally_max_order, 
            inodes_cnts, global_stats_per_lvl, DEPTH);
    #endif
        if (thread_id == lowest_lvl) {
            //num_idle_warps = atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF);
        #ifdef MODE_THEMIS
            if (num_idle_warps > 0 && globally_lowest_lvl == lowest_lvl && globally_current_max_order == locally_current_order) {
        #else
            if (num_idle_warps > 0) {
        #endif
                // Strategies 6 - Try work-stealing because this warp has INodes at the globally lowest level.
    #ifdef MODE_LB_TWOLVLBITMAPS
                Detection::TwoLvlBitmaps::Get<MIN_NUM_WARPS>(warp_id, idle_warp_id, bit1, bit2, num_idle_warps, num_warps_which_can_push); 
                if (idle_warp_id > -1) num_idle_warps = atomicSub(global_num_idle_warps, 1) - 1 - (NUM_WARPS - MIN_NUM_WARPS);
    #endif
    #ifdef MODE_LB_RANDOMIZED
                Detection::Bitmap::Get<MIN_NUM_WARPS>(idle_warp_id, bit2, num_idle_warps); 
                if (idle_warp_id > -1) num_idle_warps = atomicSub(global_num_idle_warps, 1) - 1 - (NUM_WARPS - MIN_NUM_WARPS);
    #endif
    #ifdef MODE_LB_SIMPLE         
                Detection::Stack::Get<NUM_WARPS>(idle_warp_id, global_num_idle_warps, num_idle_warps, id_stack);
    #endif
                if (idle_warp_id > -1) PushedParts::GetIthStack(gts, size_of_stack_per_warp, idle_warp_id)->TryLock();

                //if (idle_warp_id > -1) printf("hihi!!!\n");
            }

        }
        num_warps_which_can_push = __shfl_sync(ALL_LANES, num_warps_which_can_push, lowest_lvl);
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, lowest_lvl);
        idle_warp_id = __shfl_sync(ALL_LANES, idle_warp_id, lowest_lvl);

        if (idle_warp_id >= 0) assert(false);

        return;
    }


    template<uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __forceinline__ __device__ void WaitPushing(
        int warp_id, 
        unsigned int* global_num_idle_warps, unsigned int &num_idle_warps,
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp)
    {
        PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, warp_id);
        //printf("Warp %d starts to wait.. \n", warp_id);
        while(true) {
            if (stack->TryLock()) break;
            num_idle_warps = atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF) - (NUM_WARPS - MIN_NUM_WARPS);
            if (num_idle_warps == MIN_NUM_WARPS) break;
            __nanosleep(1<<10);
            //if (warp_id == 0) printf("num idle warps: %d\n", num_idle_warps);
        }
        //printf("Warp %d finishes to wait.. \n", warp_id);
    }




    template<uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ void Wait (
        int &gpart_id,
        int &busy_warp_id, 
        int warp_id, int thread_id,
        int &locally_lowest_lvl, int &globally_lowest_lvl,
        unsigned int* global_num_idle_warps, unsigned int* gpart_ids, unsigned int &num_idle_warps,
        StatisticsPerLvl* global_stats_per_lvl,
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp
#ifdef MODE_LB_TWOLVLBITMAPS
        , unsigned long long* bit1, unsigned long long* bit2
#endif
#ifdef MODE_LB_RANDOMIZED
        , unsigned long long* bit2
#endif
#ifdef MODE_LB_SIMPLE
        , Detection::Stack::IdStack* id_stack
#else
#endif
        ) 
    {
        busy_warp_id = -2;
        
        /*
        if (thread_id == 0) num_idle_warps = atomicAdd(global_num_idle_warps, 1) + 1;
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);

        // If there are enough busy warps to utilize SMs... Then hmmmmmmmmmm block으로 해야되나?
        int num_busy_warps = NUM_WARPS - num_idle_warps;
        if (num_busy_wa-rps > MIN_NUM_WARPS) {
            busy_warp_id = __shfl_sync(ALL_LANES, busy_warp_id, 0);
            locally_lowest_lvl = __shfl_sync(ALL_LANES, locally_lowest_lvl, 0);
            return;
        }

        if (gpart_id == -1) {
            if (thread_id == 0) gpart_id = atomicAdd(gpart_ids, 1);
            gpart_id = __shfl_sync(ALL_LANES, gpart_id, 0);
        }

        num_idle_warps = num_idle_warps - (NUM_WARPS - MIN_NUM_WARPS);
        */

       gpart_id = warp_id;
#ifdef MODE_LB_TWOLVLBITMAPS
        //printf("hihi!!\n");
        if (thread_id == 0) {
            //printf("hihi!!\n");
            PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, gpart_id);
            num_idle_warps = atomicAdd(global_num_idle_warps, 1) + 1;
            if (num_idle_warps < MIN_NUM_WARPS) {
                bool locked = stack->TryLock();
                //assert(locked);
                
                //Detection::TwoLvlBitmaps::Set(gpart_id, bit1, bit2);
                WaitPushing<NUM_WARPS, MIN_NUM_WARPS>(gpart_id, global_num_idle_warps, num_idle_warps, gts, size_of_stack_per_warp);
                if (num_idle_warps != MIN_NUM_WARPS) {
                    locally_lowest_lvl = stack->Top()->lvl; //(warp_id, thread_id)->lvl;
                    busy_warp_id = gpart_id;
                }
                if (num_idle_warps == MIN_NUM_WARPS) stack->lock = 0;
            }
        }
#endif
#ifdef MODE_LB_RANDOMIZED
        if (thread_id == 0) {
            PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, gpart_id);
            num_idle_warps = atomicAdd(global_num_idle_warps, 1) + 1;
            if (num_idle_warps < MIN_NUM_WARPS) {
                bool locked = stack->TryLock();
                //assert(locked);
                Detection::Bitmap::Set(gpart_id, bit2);
                WaitPushing<NUM_WARPS, MIN_NUM_WARPS>(gpart_id, global_num_idle_warps, num_idle_warps, gts, size_of_stack_per_warp);
                if (num_idle_warps != MIN_NUM_WARPS) {
                    locally_lowest_lvl = stack->Top()->lvl; //(warp_id, thread_id)->lvl;
                    busy_warp_id = gpart_id;
                }
                if (num_idle_warps == MIN_NUM_WARPS) stack->lock = 0;
            }
        }
#endif
#ifdef MODE_LB_SIMPLE
        if (thread_id == 0) {
            PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, gpart_id);
             //atomicCAS(global_num_idle_warps, 0xFFFFFFFF, 0xFFFFFFFF);
            //printf("shit warp_id: %d num_idle_warps: %d\n", warp_id, num_idle_warps);
            if (num_idle_warps == MIN_NUM_WARPS) {
                
                //num_idle_warps = atomicAdd(global_num_idle_warps, 1) + 1;
                id_stack->start = 0;
                stack->lock = 0;
            } else {
                stack->TryLock();
                Detection::Stack::Set<MIN_NUM_WARPS>(gpart_id, global_num_idle_warps, num_idle_warps-1, id_stack);
                WaitPushing<MIN_NUM_WARPS>(gpart_id, global_num_idle_warps, num_idle_warps-1, gts);
                if (num_idle_warps != MIN_NUM_WARPS) {
                    locally_lowest_lvl = stack->Top()->lvl;
                    busy_warp_id = gpart_id;
                }
            }
            //printf("shit warp_id: %d num_idle_warps: %d, busy_warp_id: %d\n", warp_id, num_idle_warps, busy_warp_id);
        }
#endif
        
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        busy_warp_id = __shfl_sync(ALL_LANES, busy_warp_id, 0);
        locally_lowest_lvl = __shfl_sync(ALL_LANES, locally_lowest_lvl, 0);
        return;
    }
}


#endif
