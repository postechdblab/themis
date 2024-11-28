#ifndef _TASKBOOK_CUH_
#define _TASKBOOK_CUH_

#include "relation.cuh"
#include "warp_status.cuh"
#include "detection.cuh"

namespace WorkSharing {

    struct Task {
        volatile int lvl_;
        volatile int start_;
        volatile int end_;
    
        __forceinline__ __device__ char* GetAttrPtr() { return ((char*) this) + 16; }    
    
        __forceinline__ __device__ bool Pop(int &lvl, int &start, int &end, int size) {
            lvl = lvl_;
            end = end_;
            start = start_;
            start_ = start + size;
            __threadfence();
            return end <= (start + size);
        }
    };

    struct TaskBook {
        unsigned long long length_;
        
        __device__ Task* GetTaskEntry(unsigned long long length) { return (Task*) (((char*) this) + 128 + length); }
    
        __device__ Task* AllocTask(int lvl, int &start, int &end, int attr_size, int threshold) {
            int num = start < end ? end - start : 0;
            if (num < threshold) return NULL;
            
            unsigned long long required_size = 16 + attr_size;
            required_size = (required_size - 1) / 16 * 16 + 16; 

            unsigned long long len = atomicAdd(&length_, required_size);
            Task* task = GetTaskEntry(len);
            
            task->lvl_ = lvl;
            task->start_ = start;
            task->end_ = end;
            //__threadfence();
            //printf("AllocTask %llx %d %d %d %d %d %d %d\n", task, lvl, start, end, 16, attr_size, required_size, len);
            start = end = 0;
            return task;
        }
    };


    struct StackEntry {
        Task* task;
    };

    struct TaskStack {
        
        int lock_;
        volatile int length_;

        __device__ StackEntry* GetStackEntry(int length) {
            StackEntry* entries_ = (StackEntry*) (((char*) this) + sizeof(TaskStack));
            return entries_ + length;
        }

        __device__ bool Push(Task* task) {
            unsigned mask = __ballot_sync(0xffffffff, task != NULL);
            if (!mask) return false;
            int num_to_push = __popc(mask);
            int thread_id = threadIdx.x % 32;
            unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));
            
            int base_length = 0;
            if (thread_id == 0) {
                while (atomicCAS(&lock_, 0, 1) == 1) {
                    __nanosleep(1 << 4);
                }
                base_length = length_;
            }
            base_length = __shfl_sync(0xffffffff, base_length, 0);
            
            if (task) {
                int length = base_length + __popc(mask & prefixlanes);
                GetStackEntry(length)->task = task;
            }
            
            if (thread_id == 0) length_ = base_length + num_to_push;
            if (thread_id == 0 || task) __threadfence();
            if (thread_id == 0) atomicCAS(&lock_, 1, 0);
            return true;
        }


        __device__ char* Pop(int &lvl, int &start, int &end, int size, bool &is_empty_after_pop) {
            is_empty_after_pop = true;
            if (length_ == 0) return NULL;
            while (atomicCAS(&lock_, 0, 1) == 1) {
                __nanosleep(1 << 4);
                if (length_ == 0) return NULL;
            }

            int length = length_;
            if (length == 0) {
                atomicCAS(&lock_, 1, 0);
                return NULL;
            }

            StackEntry* entry = GetStackEntry(length-1);
            Task* task = entry->task;

            is_empty_after_pop = task->Pop(lvl, start, end, size);
            
            if (is_empty_after_pop) {
                length_ = length-1;
                __threadfence();
            }
            is_empty_after_pop = length_ == 0;
            atomicCAS(&lock_, 1, 0);

            end = start + size < end ? start + size : end;

            return task->GetAttrPtr();
        }
    };

    __device__ char* Wait(int thread_id, int &lvl, int &start, int &end, int size,
        TaskBook* taskbook, TaskStack* taskstack, Themis::WarpsStatus* warp_status) 
    {
        lvl = -2;
        
        unsigned int num_idle_warps = 0;
        unsigned int num_working_warps = 0;

        char* attr = NULL;
        
        if (thread_id == 0) {
            num_idle_warps = warp_status->addIdleWarpNum(num_working_warps) + 1;
            do {
                if (num_working_warps == num_idle_warps) break;
                bool is_empty_after_pop = true;
                attr = taskstack->Pop(lvl, start, end, size, is_empty_after_pop);
                if (attr) {
                    num_idle_warps = warp_status->subIdleWarpNum() - 1;
                    //printf("Pop %d %d %d\n", lvl, start, end);
                    break;
                }
                warp_status->fetchWarpNum(num_idle_warps, num_working_warps);
            } while (true);
        }
        //return NULL;
        start = __shfl_sync(ALL_LANES, start, 0);
        end = __shfl_sync(ALL_LANES, end, 0);
        lvl = __shfl_sync(ALL_LANES, lvl, 0);
        return (char*) __shfl_sync(ALL_LANES, (uint64_t) attr, 0);
    }

    template<unsigned long long MIN_NUM_WARPS>
    __device__ char* Wait(int warp_id, int thread_id, TaskStack* &target_stack, int &lvl, int &start, int &end, int size,
        TaskBook* taskbook, TaskStack* taskstack,
        unsigned long long* num_buffered_inodes, unsigned long long* bit1, unsigned long long* bit2,
        Themis::WarpsStatus* warp_status)
    {
        lvl = -2;
        
        unsigned int num_idle_warps = 0;
        unsigned int num_working_warps = 0;
        target_stack = NULL;

        char* attr = NULL;
        
        if (thread_id == 0) {            
            num_idle_warps = warp_status->addIdleWarpNum(num_working_warps) + 1;
            do {
                unsigned long long n_buffered_inodes = atomicCAS(num_buffered_inodes, 0, 0);
                if (num_working_warps == num_idle_warps && n_buffered_inodes == 0) break;
                int buffer_id = -1;
                Themis::Detection::TwoLvlBitmaps::Get<MIN_NUM_WARPS>(warp_id, buffer_id, bit1, bit2, num_idle_warps);
                if (buffer_id > -1) {
                    TaskStack* target_stack = (TaskStack*) (((char*) taskstack) + buffer_id * 1024 * 64);
                    bool is_empty_after_pop = true;
                    attr = target_stack->Pop(lvl, start, end, size, is_empty_after_pop);
                    if (attr) {
                        
                        //printf("Pop %d %d %d\n", lvl, start, end);
                        unsigned long long num_to_pull = end - start;
                        n_buffered_inodes = atomicAdd(num_buffered_inodes, (unsigned long long) ((~num_to_pull) + 1));
                        //printf("n_buffered_inodes: %llu, %llu\n", n_buffered_inodes, num_to_pull);
                        num_idle_warps = warp_status->subIdleWarpNum() - 1;
                        if (!is_empty_after_pop) {
                            Themis::Detection::TwoLvlBitmaps::Set(buffer_id, bit1, bit2);
                        }
                        break;
                    }
                    
                }
                warp_status->fetchWarpNum(num_idle_warps, num_working_warps);
            } while (true);
        }
        //return NULL;
        target_stack = (TaskStack*) __shfl_sync(ALL_LANES, (uint64_t) target_stack, 0);
        start = __shfl_sync(ALL_LANES, start, 0);
        end = __shfl_sync(ALL_LANES, end, 0);
        lvl = __shfl_sync(ALL_LANES, lvl, 0);
        return (char*) __shfl_sync(ALL_LANES, (uint64_t) attr, 0);

    }
}


#endif