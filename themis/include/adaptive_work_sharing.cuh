#include "themis.cuh"
namespace Themis {

    __device__ __forceinline__ int CalculateOrder(int cnt) {
        // if cnt is zero --> order is -1
        // if cnt < 32 --> order is 0
        // if cnt < 64 --> order is 1
        return cnt <= 0 ? -1 : 32 - __clz(cnt >> 5); 
    }

    int CalculateOrderInHost(int cnt) {
        // if cnt is zero --> order is -1
        // if cnt < 32 --> order is 0
        // if cnt < 64 --> order is 1
        return cnt <= 0 ? -1 : 32 - clz(cnt >> 5); 
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

    __global__ void krnl_InitStatisticsPerLvlPtr(
        StatisticsPerLvl* stats, unsigned int num_warps, int* num_inodes_at_zero_ptr, int depth
    ) {
        int num_inodes_at_zero = *num_inodes_at_zero_ptr;

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= depth) return;

        stats[i].max_order = i == 0 ? 0 : 31;
        stats[i].num_warps = 0;

        if (i > 0 || num_inodes_at_zero == 0) return;

        stats[0].max_order = 0;
        stats[0].num_warps = 0;

        // round-robin based partitioining
        int num_inodes_of_last_block = num_inodes_at_zero % 32;
        if (num_inodes_of_last_block == 0) num_inodes_of_last_block = 32;
        
        int num_blocks = (num_inodes_at_zero + 31) / 32;
        
        //printf("num_inodes_at_zero: %d, num_blocks: %d\n", num_inodes_at_zero, num_blocks);

        int idx_case_1 = (num_blocks + num_warps -1) % num_warps;
        int num_case_0 = idx_case_1;
        int num_inodes_of_case_0 = 32 * ((num_blocks-1) / num_warps) + 32;
        int8_t order = CalculateOrder(num_inodes_of_case_0);
        
        //printf("num_inodes_at_zero: %d\n", num_inodes_at_zero);
        //printf("order case 0: %d/%d\n", order, num_case_0);
        stats[0].max_order = order;
        stats[0].sub_num_warps[order] = num_case_0; 
        
        int num_case_1 = 1;
        int num_inodes_of_case_1 = 32 * ((num_blocks-1) / num_warps) + num_inodes_of_last_block;
        order = CalculateOrder(num_inodes_of_case_1);
        //printf("order case 1: %d/%d\n", order, num_case_1);

        stats[0].sub_num_warps[order] += num_case_1; 

        int num_case_2 = num_warps - num_case_0 - 1;
        int num_inodes_of_case_2 = 32 * ((num_blocks-1) / num_warps);
        order = CalculateOrder(num_inodes_of_case_2);

        //printf("order case 2: %d/%d\n", order, num_case_2);
        stats[0].sub_num_warps[order] += num_case_2; 

        if (num_inodes_of_case_2 > 0) {
            stats[0].num_warps = num_warps;
        } else {
            stats[0].num_warps = num_case_0 + 1;
        }
    }

    __global__ void krnl_InitStatisticsPerLvl(
        StatisticsPerLvl* stats, unsigned int num_warps, int num_inodes_at_zero, int depth
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= depth) return;

        stats[i].max_order = i == 0 ? 0 : 31;
        stats[i].num_warps = 0;

        if (i > 0 || num_inodes_at_zero == 0) return;

        stats[0].max_order = 0;
        stats[0].num_warps = 0;

        // round-robin based partitioining
        int num_inodes_of_last_block = num_inodes_at_zero % 32;
        if (num_inodes_of_last_block == 0) num_inodes_of_last_block = 32;
        
        int num_blocks = (num_inodes_at_zero + 31) / 32;
        
        //printf("num_inodes_at_zero: %d, num_blocks: %d\n", num_inodes_at_zero, num_blocks);

        int idx_case_1 = (num_blocks + num_warps -1) % num_warps;
        int num_case_0 = idx_case_1;
        int num_inodes_of_case_0 = 32 * ((num_blocks-1) / num_warps) + 32;
        int8_t order = CalculateOrder(num_inodes_of_case_0);
        
        //printf("num_inodes_at_zero: %d\n", num_inodes_at_zero);
        //printf("order case 0: %d/%d\n", order, num_case_0);
        stats[0].max_order = order;
        stats[0].sub_num_warps[order] = num_case_0; 
        
        int num_case_1 = 1;
        int num_inodes_of_case_1 = 32 * ((num_blocks-1) / num_warps) + num_inodes_of_last_block;
        order = CalculateOrder(num_inodes_of_case_1);
        //printf("order case 1: %d/%d\n", order, num_case_1);

        stats[0].sub_num_warps[order] += num_case_1; 

        int num_case_2 = num_warps - num_case_0 - 1;
        int num_inodes_of_case_2 = 32 * ((num_blocks-1) / num_warps);
        order = CalculateOrder(num_inodes_of_case_2);

        //printf("order case 2: %d/%d\n", order, num_case_2);

        stats[0].sub_num_warps[order] += num_case_2; 

        if (num_inodes_of_case_2 > 0) {
            stats[0].num_warps = num_warps;
        } else {
            stats[0].num_warps = num_case_0 + 1;
        }
    }


    __global__ void krnl_InitStatisticsPerLvl(
        StatisticsPerLvl* stats, unsigned int num_warps, int num_inodes_at_zero, int depth, int num_total_active_warps, 
        int8_t order_of_case_0, int num_warps_of_case_0, int8_t order_of_case_1, int num_warps_of_case_1, int8_t order_of_case_2, int num_warps_of_case_2)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i >= depth) return;

        stats[i].max_order = i == 0 ? 0 : 31;
        stats[i].num_warps = 0;
        

        if (i != 0 || num_inodes_at_zero == 0) return;
        stats[0].num_warps = num_total_active_warps;
        stats[0].max_order = order_of_case_0;
        stats[0].sub_num_warps[order_of_case_0] = num_warps_of_case_0;
        stats[0].sub_num_warps[order_of_case_1] += num_warps_of_case_1;
        stats[0].sub_num_warps[order_of_case_2] += num_warps_of_case_2;
        //printf("stats[0].max_order: %d, num_warps: %d, num_inodex_at_zero: %d\n", stats[0].max_order, stats[0].num_warps, num_inodes_at_zero);
    }

    void InitStatisticsPerLvl(StatisticsPerLvl* stats, unsigned int num_warps, int num_inodes_at_zero, int depth) {
        krnl_InitStatisticsPerLvl<<<1,128>>>(stats, num_warps, num_inodes_at_zero, depth);
    }

    void InitStatisticsPerLvlPtr(StatisticsPerLvl* stats, unsigned int num_warps, int* num_inodes_at_zero_ptr, int depth) {
        krnl_InitStatisticsPerLvlPtr<<<1,128>>>(stats, num_warps, num_inodes_at_zero_ptr, depth);
    }

    void InitStatisticsPerLvl(StatisticsPerLvl* &device_stats, unsigned int num_warps) {
        cudaMalloc((void**)&device_stats, sizeof(StatisticsPerLvl) * 128);
        cudaMemset(device_stats, 0, sizeof(StatisticsPerLvl) * 128);
        //krnl_InitStatisticsPerLvl<<<1,128>>>(device_stats, num_warps, depth);
    }

    void PrintStatisticsPerLvl(StatisticsPerLvl* device_stats, const char* krnl_name, int depth, unsigned int num_warps) {
        //return;
        StatisticsPerLvl host_stats[128];
        cudaMemcpy(host_stats, device_stats, sizeof(StatisticsPerLvl) * depth, cudaMemcpyDeviceToHost);
        printf("num warps per lvl: \n");
        for (int i = 0; i < depth; ++i) {
            printf("\t%d (", host_stats[i].num_warps);
            for (int j = 0; j < 16; ++j) {
                printf("%d ", host_stats[i].sub_num_warps[j]);
            }
            printf(")\n");
            //printf("%s %d # push operations: %llu\n", krnl_name, i, host_stats[i].num_ws);
            //printf("%s_%d stats: %d, %d, %llu, %llu, %llu, %lf, %lf\n", 
            //printf("%s lvl, max # inodes, # work stealings, # inodes, # nodes, # inodes / # nodes, 32 * # num ws / # num nodes : %d, %d, %llu, %llu, %llu, %lf, %lf\n", 
                //krnl_name, i, host_stats[i].max_inodes_cnt, host_stats[i].num_ws, host_stats[i].num_inodes_at_that_time, 
                //host_stats[i].num_nodes, (double)host_stats[i].num_inodes_at_that_time / (double)host_stats[i].num_nodes, 
                //64.0 * (double)host_stats[i].num_ws / (double)host_stats[i].num_nodes);
        }
        printf("\n");
        //cudaMemset(device_stats, 0, sizeof(StatisticsPerLvl) * depth);
        //krnl_InitStatisticsPerLvl<<<1,64>>>(device_stats, num_warps);
    }

    struct LocalLevelAndOrderInfo {
        int8_t locally_lowest_lvl;
        int8_t globally_lowest_lvl;
        int8_t locally_max_order;
        int8_t globally_max_order;
        int num_nodes_at_locally_lowest_lvl;
        unsigned int num_warps_which_can_push;

        __device__ void print(int lvl) {
            int warp_id =(blockIdx.x * blockDim.x + threadIdx.x) / 32;
            printf("LvlAndOrder %d, %d: %d, %d, %d, %d, %d\n", warp_id, lvl, num_nodes_at_locally_lowest_lvl, locally_lowest_lvl, locally_max_order, globally_lowest_lvl, globally_max_order);
        }
    };

    __device__ __forceinline__ void FetchLvlAndOrder(
        int thread_id,
        LocalLevelAndOrderInfo &local_info,
        StatisticsPerLvl* global_stats_per_lvl)
    {
        int8_t locally_lowest_lvl = local_info.locally_lowest_lvl;
        int8_t locally_max_order = local_info.locally_max_order;
        int8_t &globally_lowest_lvl = local_info.globally_lowest_lvl;
        int8_t &globally_max_order = local_info.globally_max_order;
        if (thread_id == 0) {
            
            while (locally_lowest_lvl > globally_lowest_lvl) {
                unsigned int num_warps_at_globally_lowest_lvl = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].num_warps, 0xFFFFFFFF, 0xFFFFFFFF);
                if (num_warps_at_globally_lowest_lvl > 0) break;
                ++globally_lowest_lvl;
                //printf("GLVL from %d -> %d\n", globally_lowest_lvl-1, globally_lowest_lvl);
            }
            //assert(globally_lowest_lvl <= locally_lowest_lvl);
            if (locally_lowest_lvl == globally_lowest_lvl) {
                globally_max_order = (int8_t) atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].max_order,0xFFFFFFFF, 0xFFFFFFFF);
                local_info.num_warps_which_can_push = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].sub_num_warps[globally_max_order],0xFFFFFFFF, 0xFFFFFFFF);
                int8_t new_globally_max_order = globally_max_order;
                while (local_info.num_warps_which_can_push == 0) {
                    --new_globally_max_order;
                    local_info.num_warps_which_can_push = 
                        atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].sub_num_warps[new_globally_max_order],0xFFFFFFFF, 0xFFFFFFFF);
                }
                //if (new_globally_max_order < locally_max_order) {
                    //printf("wrong... at lvl %d, gmo: %d, lmo: %d\n", globally_lowest_lvl, new_globally_max_order, locally_max_order);
                //    assert(false);
                //}
                //assert(new_globally_max_order >= locally_max_order);
                if (globally_max_order > new_globally_max_order) {
                    //printf("order from %d -> %d\n", globally_max_order, new_globally_max_order);
                    atomicMin(&global_stats_per_lvl[globally_lowest_lvl].max_order, new_globally_max_order);
                }
                globally_max_order = new_globally_max_order;
            }
        }
        local_info.num_warps_which_can_push = __shfl_sync(ALL_LANES, local_info.num_warps_which_can_push, 0);
        globally_lowest_lvl = __shfl_sync(ALL_LANES, globally_lowest_lvl, 0);
        globally_max_order = __shfl_sync(ALL_LANES, globally_max_order, 0);
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
        if (thread_id == 0) {
            // Update the globally lowest lvel
            while (true) {
                num_warps_at_globally_lowest_lvl = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].num_warps, 0xFFFFFFFF, 0xFFFFFFFF);
                if (num_warps_at_globally_lowest_lvl > 0) break;
                if (++globally_lowest_lvl >= depth) break;
            }
            // Update the maximum order if lowest_lvl is the globally_lowest_lvl
            if (lowest_lvl == globally_lowest_lvl) {    
                globally_current_max_order = atomicCAS(&global_stats_per_lvl[globally_lowest_lvl].max_order,0xFFFFFFFF, 0xFFFFFFFF);
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
        globally_current_max_order = __shfl_sync(ALL_LANES, globally_current_max_order, lowest_lvl);
    }

    __device__ __forceinline__ void chooseNextIntervalAfterPush(unsigned int &interval, LocalLevelAndOrderInfo &local_info, int num_warps, int num_idle_warps, bool is_allowed, unsigned int max_interval) {

        interval = (interval >= 32 || interval == 0) ? 1 : 2 * interval;
        unsigned int num_busy_warps = num_warps - num_idle_warps; //local_info.num_warps_which_can_push;
        interval = (32 - __clz(num_warps / num_idle_warps));
    }

    __device__ __forceinline__ void chooseNextIntervalAfterPush(unsigned int &interval, unsigned int num_warps, unsigned int num_idle_warps, bool is_allowed, unsigned int max_interval) {
        unsigned int num_busy_warps = num_warps - num_idle_warps;
        interval = (32 - __clz(num_busy_warps / num_idle_warps));        
    }

    __device__ bool isPushingAllowed(
        int thread_id,  
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps, unsigned int &num_warps,
        LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl)
    {
        FetchLvlAndOrder(thread_id, local_info, global_stats_per_lvl);
        bool is_allowed = false;
        if (thread_id == 0) {
            warp_status->fetchWarpNum(num_idle_warps, num_warps);
            //printf("glvl %d, llvl %d, gmo %d, lmo %d\n", local_info.globally_lowest_lvl, local_info.locally_lowest_lvl, local_info.globally_max_order, local_info.locally_max_order);
            is_allowed = (num_idle_warps > 0) &&
                (local_info.globally_lowest_lvl == local_info.locally_lowest_lvl) && 
                ((local_info.globally_max_order == local_info.locally_max_order) || 
                    ((local_info.globally_max_order == local_info.locally_max_order + 1) && (num_idle_warps >= 2 * local_info.num_warps_which_can_push)))
                ;
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        num_warps = __shfl_sync(ALL_LANES, num_warps, 0);
        return __shfl_sync(ALL_LANES, is_allowed, 0);
    }

    __device__ bool isPushingAllowed(
        int thread_id,
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps, unsigned int &num_warps)
    {
        bool is_allowed = false;
        if (thread_id == 0) {
            warp_status->fetchWarpNum(num_idle_warps, num_warps);
            is_allowed = num_idle_warps > 0;
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        num_warps = __shfl_sync(ALL_LANES, num_warps, 0);
        return __shfl_sync(ALL_LANES, is_allowed, 0);
    }

    // TwoLvlBitmaps based idle warp detection
    template<int DEPTH, uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ __forceinline__ void FindIdleWarp(
        int &idle_warp_id, int warp_id, int thread_id,
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps, unsigned int num_warps,
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp
        , unsigned long long* bit1, unsigned long long* bit2)
    {
        idle_warp_id = -1;
        if (thread_id == 0) {
            Detection::TwoLvlBitmaps::Get<MIN_NUM_WARPS>(warp_id, idle_warp_id, bit1, bit2, num_idle_warps);
            if (idle_warp_id > -1) {
                num_idle_warps = warp_status->subIdleWarpNum() - 1;
                PushedParts::GetIthStack(gts, size_of_stack_per_warp, idle_warp_id)->TryLock();
            }
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        idle_warp_id = __shfl_sync(ALL_LANES, idle_warp_id, 0);
    }

    // Stack based idle warp detectino. 
    template<int DEPTH, uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ __forceinline__ void FindIdleWarp(
        int &idle_warp_id, int warp_id, int thread_id,
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps, unsigned int num_warps,
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp
        , Detection::Stack::IdStack* id_stack)
    {
        idle_warp_id = -1;
        if (thread_id == 0) {        
            //printf("find idle warp %d, %d\n", warp_id, idle_warp_id);
            
            Detection::Stack::Get<MIN_NUM_WARPS>(idle_warp_id, warp_status, num_idle_warps, id_stack);
            if (idle_warp_id > -1) {
                PushedParts::GetIthStack(gts, size_of_stack_per_warp, idle_warp_id)->TryLock();
            }
            //printf("~find idle warp %d, %d\n", warp_id, idle_warp_id);
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        idle_warp_id = __shfl_sync(ALL_LANES, idle_warp_id, 0);
    }



    template<uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __forceinline__ __device__ void WaitPushing(
        int warp_id, 
        WarpsStatus* warp_status,
        //unsigned int* global_num_idle_warps,  unsigned int* gpart_ids,
        unsigned int &num_idle_warps, unsigned int &num_working_warps, 
        PushedParts::PushedPartsStack* gts, int size_of_stack_per_warp)
    {
        PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, warp_id);
        unsigned i = 0;
        unsigned try_lock_interval = 1;
        unsigned next_try_lock = 0;
        unsigned check_termination_interval = 1;
        unsigned next_check_termination = 0;
        while(true) {
            if (next_try_lock <= i) {
                if (stack->TryLock()) break;
                next_try_lock = i + try_lock_interval;
            }
            if (next_check_termination <= i) {
                warp_status->fetchWarpNum(num_idle_warps, num_working_warps);
                if (num_idle_warps == num_working_warps) break;
                unsigned int num_busy_warps = num_working_warps - num_idle_warps;
                if (num_busy_warps < 64) {
                    try_lock_interval = 16;
                    check_termination_interval = 1;
                } else if (num_busy_warps < 256) {
                    try_lock_interval = 8;
                    check_termination_interval = 8;
                } else {
                    try_lock_interval = 4;
                    check_termination_interval = 64;
                }
                next_check_termination = i + try_lock_interval;
            } 
            __nanosleep(1 << 10);
            ++i;
        }
    }

    template<uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ void Wait (
        int &gpart_id,
        int &busy_warp_id, 
        int warp_id, int thread_id,
        int &lowest_lvl,
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps,
        StatisticsPerLvl* global_stats_per_lvl,
        PushedParts::PushedPartsStack* gts, size_t size_of_stack_per_warp
        , unsigned long long* bit1, unsigned long long* bit2) 
    {
        busy_warp_id = -2;
        unsigned int num_working_warps = 0;
        if (thread_id == 0) num_idle_warps = warp_status->addIdleWarpNum(num_working_warps) + 1;
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        if (num_idle_warps == NUM_WARPS) return;
        num_working_warps = __shfl_sync(ALL_LANES, num_working_warps, 0);
        if (num_idle_warps == num_working_warps) return;
        if (gpart_id >= MIN_NUM_WARPS) return;
        if (thread_id == 0) {
            if (num_idle_warps != num_working_warps) {
                PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, gpart_id);
                bool locked = stack->TryLock();
                Detection::TwoLvlBitmaps::Set(gpart_id, bit1, bit2);
                WaitPushing<NUM_WARPS, MIN_NUM_WARPS>(gpart_id, 
                    warp_status,
                    num_idle_warps, num_working_warps,  gts, size_of_stack_per_warp);
                
                if (num_idle_warps != num_working_warps) {
                    lowest_lvl = stack->Top()->lvl; //(warp_id, thread_id)->lvl;
                    busy_warp_id = gpart_id;
                } else {
                    stack->lock = 0;
                    
                } 
            }
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        busy_warp_id = __shfl_sync(ALL_LANES, busy_warp_id, 0);
        lowest_lvl = __shfl_sync(ALL_LANES, lowest_lvl, 0);
    }

    template<uint64_t NUM_WARPS, uint64_t MIN_NUM_WARPS>
    __device__ void Wait (
        int &gpart_id,
        int &busy_warp_id, 
        int warp_id, int thread_id,
        int &lowest_lvl,
        WarpsStatus* warp_status,
        unsigned int &num_idle_warps,
        StatisticsPerLvl* global_stats_per_lvl,
        PushedParts::PushedPartsStack* gts, size_t size_of_stack_per_warp,
        Detection::Stack::IdStack* id_stack) 
    {
        busy_warp_id = -2;
        unsigned int num_working_warps = 0;
        if (thread_id == 0) warp_status->fetchWarpNum(num_idle_warps, num_working_warps);
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        num_working_warps = __shfl_sync(ALL_LANES, num_working_warps, 0);

        if (((num_idle_warps + 1) == NUM_WARPS) || gpart_id >= MIN_NUM_WARPS) {
            if (thread_id == 0)  {
                while (0 != atomicCAS(&id_stack->lock, 0, 1)) __nanosleep(1 << 4); 
                num_idle_warps = warp_status->addIdleWarpNum(num_working_warps) + 1;
                atomicCAS(&id_stack->lock, 1, 0);
            }
            num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
            lowest_lvl = __shfl_sync(ALL_LANES, lowest_lvl, 0);
            return;
        }
        if (thread_id == 0) {
            PushedParts::PushedPartsStack* stack = PushedParts::GetIthStack(gts, size_of_stack_per_warp, gpart_id);
            stack->TryLock();
            //printf("wait %d, %d, %d:q\n", gpart_id, num_idle_warps, num_working_warps);
            Detection::Stack::Set<NUM_WARPS>(gpart_id, warp_status, num_idle_warps, id_stack);
            //printf("~wait %d, %d, %d:q\n", gpart_id, num_idle_warps, num_working_warps);
            WaitPushing<NUM_WARPS, MIN_NUM_WARPS>(gpart_id, 
                    warp_status, num_idle_warps, num_working_warps,  gts, size_of_stack_per_warp);
            if (num_idle_warps != num_working_warps) {
                lowest_lvl = stack->Top()->lvl; //(warp_id, thread_id)->lvl;
                busy_warp_id = gpart_id;
            } else {
                stack->lock = 0;
                
            } 
        }
        num_idle_warps = __shfl_sync(ALL_LANES, num_idle_warps, 0);
        busy_warp_id = __shfl_sync(ALL_LANES, busy_warp_id, 0);
        lowest_lvl = __shfl_sync(ALL_LANES, lowest_lvl, 0);
    }


    namespace WorkloadTracking {

        __device__ __forceinline__ 
        void UpdateWorkloadSizeOfIdleWarpAfterPush(int thread_id, 
            int lvl_to_push,
            int num_to_push,
            StatisticsPerLvl* global_stats_per_lvl) 
        {
            if (thread_id == 0) {
                int8_t new_order = CalculateOrder(num_to_push);
                atomicAdd(&global_stats_per_lvl[lvl_to_push].sub_num_warps[new_order], 1);
                atomicAdd(&global_stats_per_lvl[lvl_to_push].num_warps, 1);
            }
        }

        __device__ __forceinline__
        void UpdateWorkloadSizeOfBusyWarpAfterPush(int thread_id,
            unsigned mask_1,
            int new_num_nodes_at_locally_lowest_lvl, int new_local_lowest_lvl, int8_t new_local_max_order,
            LocalLevelAndOrderInfo &local_info,
            StatisticsPerLvl* global_stats_per_lvl) 
        {
            if (thread_id == 0) {
                if (local_info.locally_lowest_lvl != new_local_lowest_lvl) {
                    if (mask_1 != 0) {
                        atomicAdd(&global_stats_per_lvl[new_local_lowest_lvl].sub_num_warps[new_local_max_order], 1);
                        atomicAdd(&global_stats_per_lvl[new_local_lowest_lvl].num_warps, 1);
                    }
                } else {
                    atomicAdd(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[new_local_max_order], 1);
                }
                atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[local_info.locally_max_order], 1);
                if (local_info.locally_lowest_lvl != new_local_lowest_lvl) atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_warps, 1);
            }

            local_info.locally_lowest_lvl = new_local_lowest_lvl;
            local_info.locally_max_order = new_local_max_order;
            local_info.num_nodes_at_locally_lowest_lvl = new_num_nodes_at_locally_lowest_lvl;
        }

        __device__ __forceinline__ void UpdateWorkloadSizeAtZeroLvl(
            int thread_id, unsigned &loop, LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl)
        {   
            local_info.num_nodes_at_locally_lowest_lvl -= 32;
            if (local_info.num_nodes_at_locally_lowest_lvl < CalculateLowerBoundOfOrder(local_info.locally_max_order)) {
                int8_t old_max_order = local_info.locally_max_order;
                local_info.locally_max_order = CalculateOrder(local_info.num_nodes_at_locally_lowest_lvl);
                //assert(old_max_order > local_info.locally_max_order);        
                if (thread_id == 0) {
                    if (local_info.locally_max_order >= 0) atomicAdd(&global_stats_per_lvl[0].sub_num_warps[local_info.locally_max_order], 1);
                    unsigned int sub_num_warps = atomicSub(&global_stats_per_lvl[0].sub_num_warps[old_max_order], 1);
                    //printf("sub_num_warps %d %d %d\n", 0, old_max_order, sub_num_warps-1);
                }
                if (local_info.locally_max_order == -1) --loop;
            }
        }

        __device__ __forceinline__ void UpdateWorkloadSizeAtLoopLvl(
            int thread_id, int lvl, unsigned &loop, Range &range, Range &range_cached, unsigned &mask_1,
            LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl)
        {
            if (mask_1 & (~(0xFFFFFFFFu << lvl))) return;
            if (local_info.locally_max_order < 0) {
                if (!(mask_1 & (0x1u << lvl))) {
                    --loop;
                    return;
                }  
                int cnt = range.start < range.end ? range.end - range.start : 0;
                cnt += range_cached.start < range_cached.end ? ((range_cached.end - range_cached.start - 1) >> 5) + 1 : 0;
                PrefixSumDown(thread_id, cnt);
                cnt = __shfl_sync(ALL_LANES, cnt, 0);
                local_info.num_nodes_at_locally_lowest_lvl = cnt;
                int8_t new_max_order = CalculateOrder(cnt);
                int8_t new_locally_lowest_lvl = lvl;
                if (thread_id == 0) {
                    atomicAdd(&global_stats_per_lvl[lvl].num_warps, 1);
                    atomicAdd(&global_stats_per_lvl[lvl].sub_num_warps[new_max_order], 1);
                    atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_warps, 1);
                }
                local_info.locally_lowest_lvl = new_locally_lowest_lvl;
                local_info.locally_max_order = new_max_order;
                return;
            }
            local_info.num_nodes_at_locally_lowest_lvl -= 32;
            if (local_info.num_nodes_at_locally_lowest_lvl < CalculateLowerBoundOfOrder(local_info.locally_max_order)) {
                int8_t new_max_order = CalculateOrder(local_info.num_nodes_at_locally_lowest_lvl);            
                if (thread_id == 0) {
                    if (new_max_order >= 0) atomicAdd(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[new_max_order], 1);
                    atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[local_info.locally_max_order], 1);
                }
                local_info.locally_max_order = new_max_order;
                if (new_max_order == -1) --loop;
            }
        }

        __device__ __forceinline__ void UpdateWorkloadSizeAtIfLvl(
            int thread_id, int lvl, unsigned &loop, int inodes_cnts, unsigned &mask_1,
            LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl)
        {
            if (mask_1 & (~(0xFFFFFFFFu << lvl))) return;
            if (local_info.locally_max_order < 0) {
                if (!(mask_1 & (0x1u << lvl))) {
                    --loop;
                    return;
                }  
                local_info.num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, lvl);
                if (thread_id == 0) {
                    atomicAdd(&global_stats_per_lvl[lvl].num_warps, 1);
                    atomicAdd(&global_stats_per_lvl[lvl].sub_num_warps[0], 1);
                    atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_warps, 1);
                }
                local_info.locally_lowest_lvl = lvl;
                local_info.locally_max_order = 0;
                return;
            }
            if (lvl != local_info.locally_lowest_lvl) return;
            local_info.num_nodes_at_locally_lowest_lvl -= 32;
            if (local_info.num_nodes_at_locally_lowest_lvl < CalculateLowerBoundOfOrder(local_info.locally_max_order)) {
                int8_t new_max_order = CalculateOrder(local_info.num_nodes_at_locally_lowest_lvl);            
                if (thread_id == 0) {
                    if (new_max_order >= 0) atomicAdd(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[new_max_order], 1);
                    atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].sub_num_warps[local_info.locally_max_order], 1);
                }
                local_info.locally_max_order = new_max_order;
                if (new_max_order == -1) --loop;
            }
        }

        __device__ __forceinline__ void InitLocalWorkloadSizeAtZeroLvl(int num_nodes, LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl) {
            local_info.num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, num_nodes, 0);
            local_info.globally_max_order = global_stats_per_lvl[0].max_order;
            local_info.globally_lowest_lvl = 0;
            if (local_info.num_nodes_at_locally_lowest_lvl > 0) {
                local_info.locally_max_order = Themis::CalculateOrder(local_info.num_nodes_at_locally_lowest_lvl);
                local_info.locally_lowest_lvl = 0;
            } else {
                local_info.locally_max_order = -1;
                local_info.locally_lowest_lvl = -1;
            }
        }

        __device__ __forceinline__ void InitLocalWorkloadSize(int lvl, int num_nodes, LocalLevelAndOrderInfo &local_info, StatisticsPerLvl* global_stats_per_lvl) {
            local_info.num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, num_nodes, lvl);
            local_info.locally_max_order = Themis::CalculateOrder(local_info.num_nodes_at_locally_lowest_lvl);
            local_info.locally_lowest_lvl = lvl;
            local_info.globally_max_order = local_info.locally_max_order + 1;
            local_info.globally_lowest_lvl = lvl;
        }
    }
}


