#ifndef _RANGE_CUH_
#define _RANGE_CUH_    
struct Range {
    int start = 0;
    int end = 0;
    
    __device__ __host__ Range() {
        start = end = 0;
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

#endif