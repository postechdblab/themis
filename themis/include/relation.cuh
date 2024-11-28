#ifndef _RELATION_CUH__
#define _RELATION_CUH__

#include <stdint.h>
#include <stdio.h>
#include <cassert>

#define ERROR(msg) \
    fprintf(stderr, "ERROR: %s\n", msg); \
    fprintf(stderr, "Line %i of function %s in file %s\n", __LINE__, __func__, __FILE__); \
    exit(EXIT_FAILURE);


#define ALL_LANES 0xffffffff

__device__ unsigned int get_smid() {
     unsigned int ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}


template<typename T>
struct GpuPrimitiveType {

    T val_at_host;
    T* val_at_device = NULL;

    GpuPrimitiveType() {
        cudaMalloc((void**)&val_at_device, sizeof(T));
    }

    ~GpuPrimitiveType() {
        cudaFree(val_at_device);
    }

    T& operator=(T other) {
        val_at_host = other;
        return val_at_host;
    }

    T get() {
        return val_at_host;
    }

    void HtoD() {
        cudaMemcpy(val_at_device, &val_at_host, sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void DtoH() {
        cudaMemcpy(&val_at_host, val_at_device, sizeof(T), cudaMemcpyDeviceToHost);
    }

    T* ptrAtDevice() {
        return val_at_device;
    }
};

template<typename T>
struct GpuArrayType {

    T* val_at_host = NULL;
    T* val_at_device = NULL;
    size_t size = 0;

    GpuArrayType() {
        val_at_host = NULL;
        val_at_device = NULL;    
    }

    ~GpuArrayType() {
        if (val_at_host != NULL) free(val_at_host);
        if (val_at_device != NULL) cudaFree(val_at_device);
    }

    void resize(size_t _size) {
        if (val_at_host != NULL) free(val_at_host);
        if (val_at_device != NULL) cudaFree(val_at_device);
        val_at_host = val_at_device = NULL;
        val_at_host = (T*) malloc(sizeof(T) * _size);
        cudaMalloc((void**) &val_at_device, sizeof(T) * _size);
        size = _size;
    }

    void setAll(T v) {
        for (size_t i = 0; i < size; ++i) {
            val_at_host[i] = v;
        }
    }

    T* ptrAtDevice() {
        return val_at_device;
    }

    T& operator[] (size_t idx) {
        return val_at_host[idx];
    }

    void HtoD() {
        cudaMemcpy(val_at_device, val_at_host, sizeof(T) * size, cudaMemcpyHostToDevice);
    }

    void DtoH() {
        cudaMemcpy(val_at_host, val_at_device, sizeof(T) * size, cudaMemcpyDeviceToHost);
    }

};


__device__ int dummyLoop (int a, int num) {
    int result = 1;
    for (int i = 0; i < num; ++i) {
        result *= a;
    }
    return result;
}

struct str_t {
    char* start;
    char* end;

    bool operator == (const str_t &b)
	{
        str_t &a = *this;
        int lena = end - start;
        int lenb = b.end - b.start;
        if ( lena != lenb )
            return false;
        char* c = a.start;
        char* d = b.start;
        for ( ; c < a.end; c++, d++ ) {
            if ( *c != *d ) 
                return false;
        }
        return true;
	}
};

void stringPrint ( str_t str, FILE* fout=stdout, int nchars=10 ) {
    while (str.start < str.end) {
        fputc (*str.start, fout );
        str.start++;
    }
}

struct str_offs {
    size_t start;
    size_t end;
};

void stringPrint ( char* chrs, str_offs s, FILE* fout=stdout, int nchars=10 ) {
    int n = s.end - s.start;
    int i = 0;
    int p = 0; 
 
    for (p=0; p<nchars-n; p++)
        fputc (' ', fout );
        
    for (; p<nchars; p++,i++)
        fputc ( chrs[s.start+i], fout );
}

// intialize an array as used e.g. for join hash tables
template<typename T>
__global__ void initArray ( T* array, T value, int num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
        array[i] = value;
    }
}

__global__ void initArray ( unsigned long long* array, int value, int num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
        array[i] = (unsigned long long) value;
    }
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ static double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __forceinline__ str_t stringScan ( const size_t* offs, char* chrs, int ix ) {
    str_t s;
    size_t start = offs[ix];
    s.start = chrs + start;
    size_t end = offs[ix + 1];
    s.end = chrs + end;
    return s;
}

__device__ __forceinline__ str_t stringScan1 ( const size_t* offs, char* chrs, int ix ) {
    str_t s;
    size_t start = offs[ix];
    s.start = chrs + start;
    size_t end = offs[ix + 1];
    s.end = chrs + end;
    return s;
}

__device__ __forceinline__ str_t stringScan2 ( const size_t* offs, char* chrs, int ix ) {
    str_t s;
    size_t start = offs[ix];
    s.start = chrs + start;
    size_t end = offs[ix + 1];
    s.end = chrs + end;
    return s;
}

__device__ str_offs toStringOffset ( char* chrs, str_t s ) {
    str_offs o;
    o.start = s.start - chrs;
    o.end = s.end - chrs;
    return o;
}

__device__ str_t stringConstant ( const char* chars, int len ) {
    str_t s;
    s.start = const_cast<char*>(chars);
    s.end = const_cast<char*>(chars) + len;
    return s;

}

__device__ str_t __shfl_sync ( unsigned lanemask, str_t v, int sourceLane ) {
    str_t res;
    res.start = (char*) __shfl_sync ( lanemask, (uint64_t) v.start, sourceLane );
    res.end   = (char*) __shfl_sync ( lanemask, (uint64_t) v.end,   sourceLane );
    return res;
}

__device__ bool stringEquals ( const str_t &a, const str_t &b ) {
    int lena = a.end - a.start;
    int lenb = b.end - b.start;
    if ( lena != lenb )
        return false;
    //if (a.start == b.start) return true;
    char* c = a.start;
    char* d = b.start;
    for ( ; c < a.end; c++, d++ ) {
        if ( *c != *d ) 
            return false;
    }
    return true;
}

__device__ bool stringEquals ( const str_t &a, const str_t &b, int &iteration ) {
    //iteration = 0;
    int lena = a.end - a.start;
    int lenb = b.end - b.start;
    if ( lena != lenb )
        return false;
    //if (a.start == b.start) return true;
    char* c = a.start;
    char* d = b.start;
    for ( ; c < a.end; c++, d++ ) {
        if ( *c != *d ) 
            return false;
        ++iteration;
    }
    return true;
}


__device__ bool stringEqualsPushdown ( bool active, str_t a, str_t b ) {
    unsigned warplane = threadIdx.x % 32;
    bool equal = active;
    if ( equal ) { 
        equal = ( (a.end - a.start) == (b.end - b.start) );
    }
    unsigned cmpTodoMask = __ballot_sync ( ALL_LANES, equal );
    while ( cmpTodoMask > 0 ) {
        unsigned strLane = ( __ffs ( cmpTodoMask ) - 1 );
        cmpTodoMask -= ( 1 << strLane );
        str_t strA = __shfl_sync ( ALL_LANES, a, strLane);
        str_t strB = __shfl_sync ( ALL_LANES, b, strLane);
        bool currEqual = true;
        char* chrA = strA.start + warplane;
        char* chrB = strB.start + warplane;
        while ( chrA < strA.end && currEqual ) {
            currEqual &= __all_sync ( ALL_LANES, (*chrA) == (*chrB) );
            chrA += 32;
            chrB += 32;
        }
        if ( warplane == strLane ) {
            equal = currEqual;
        }
    }
    return equal || (!active);
}


__device__ str_t stringSubstring ( str_t str, int from, int fr ) {
    str_t res;
    // todo: throw error if for is negative
    res.start = str.start + from - 1;
    res.end = res.start + fr;
    if ( res.start < str.start )
        res.start = str.start;
    if ( res.end > str.end )
        res.end = str.end;
    return res;
}

__inline__ __device__ bool cmpLike ( char c, char l ) {
    return ( c == l ) || ( l == '_' );
}

__device__ bool stringLikeCheck ( str_t string, str_t like ) {
    char *sPos, *lPos, *sTrace, *lTrace;
    char *lInStart = like.start;
    char *lInEnd   = like.end;
    char *sInStart = string.start;
    char *sInEnd   = string.end;

    // prefix 
    if ( *like.start != '%' ) { 
        sPos = string.start;
        lPos = like.start;
        for ( ; lPos < like.end && sPos < string.end && (*lPos) != '%'; ++lPos, ++sPos ) {
            if ( !cmpLike ( *sPos, *lPos ) )
                return false;
        }
        lInStart = lPos; 
        sInStart = sPos; 
    }
    
    // suffix 
    if ( *(like.end-1) != '%' ) {
        sPos = string.end-1;
        lPos = like.end-1;
        for ( ; lPos >= like.start && sPos >= string.start && (*lPos) != '%'; --lPos, --sPos ) {
            if ( !cmpLike ( *sPos, *lPos ) )
                return false;
        }
        lInEnd = lPos;
        sInEnd = sPos+1; // first suffix char 
    }

    // infixes 
    if ( lInStart < lInEnd ) {
        lPos = lInStart+1; // skip '%'
        sPos = sInStart;
        while ( sPos < sInEnd && lPos < lInEnd ) { // loop 's' string
            lTrace = lPos;
            sTrace = sPos;
            while ( cmpLike ( *sTrace, *lTrace ) && sTrace < sInEnd ) { // loop infix matches
                ++lTrace;
                if ( *lTrace == '%' ) {
                    lPos = ++lTrace;
                    sPos = sTrace;
                    break;
                }
                ++sTrace; 
            }
            ++sPos;
        }
    }
    return lPos >= lInEnd;
}




#define HASH_EMPTY 0xffffffffffffffff
#define HASH_MAX   0x7fffffffffffffff


__device__ __forceinline__ uint64_t hash(uint64_t key)
{
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return key & (HASH_MAX);
}


__device__ __forceinline__ uint64_t stringHash ( str_t s ) {
    uint64_t h = 1;
    unsigned char rem = 0;
    for ( char* c = s.start; c < s.end; ++c ) {
        char exChr = *(c);
            h *= exChr; 
        rem++;
        // apply hash function every 8 byte
        if ( rem%8 == 0 ) {
            h = hash ( h );
            rem = 0;
        }
    }
    return h;
}


__device__ uint64_t stringHashPushDown ( bool active, str_t s ) {
    uint64_t hashResult = 1;
    unsigned warplane = threadIdx.x % 32;
    unsigned hashTodoMask = __ballot_sync ( ALL_LANES, active );
    while ( hashTodoMask > 0 ) {
        unsigned strLane = ( __ffs ( hashTodoMask ) - 1 );
        hashTodoMask -= ( 1 << strLane );
        str_t hs = __shfl_sync ( ALL_LANES, s, strLane);
    	char* c = hs.start + warplane;
    	uint32_t hsub = 1;
        while ( c < hs.end ) {
	    hsub *= (*c);
	    c+=32;
        }
	hsub = hash ( hsub );
	uint64_t hash;
        hash  = (uint64_t) __ballot_sync ( ALL_LANES, hsub & 1u );
	hash += (uint64_t) __ballot_sync ( ALL_LANES, hsub & 2u ) << 32;
	if ( warplane == strLane ) {
	    hashResult = hash;
        }
    }
    return hashResult;
}

template <typename T>
struct unique_ht {
    uint64_t hash;
    T payload;
};

// intialize an array as used e.g. for join hash tables
template <typename T>
__global__ void initUniqueHT ( unique_ht<T>* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].hash = HASH_EMPTY;
    }
}

template <typename T>
__device__ void hashBuildUnique ( unique_ht<T>* hash_table, int ht_size, uint64_t hash, T* payload ) {
    int org_location = hash % ht_size;
    uint32_t location = org_location;
    unique_ht<T>* elem;

    while ( true )  {
        elem = &(hash_table[location]);
        unsigned long long probe_hash = atomicCAS( (unsigned long long*) &(elem->hash), HASH_EMPTY, (unsigned long long)hash );
        if(probe_hash == HASH_EMPTY) {
            elem->payload = *payload;
	    return;
        }
        location = (location + 1) % ht_size;
        if(location == org_location) {
            printf ( "build on full hash table, location: %i\n", location );
            return;
        }
    }
}

template <typename T>
__device__ bool hashProbeUnique ( unique_ht<T>* hash_table, int ht_size, uint64_t hash, int& numLookups, int& location) { //,T** payload ) {
    //int location;
    while ( numLookups < ht_size )  {
        location = ( hash + numLookups ) % ht_size;
        unique_ht<T>& elem = hash_table[location];
        numLookups++;
        if ( elem.hash == hash ) {
            //*payload = &elem.payload;
            return true;
        } else if ( elem.hash == HASH_EMPTY ) {
            return false;
        }
    }
    //printf ( "probing full hash table - num lookups: %i\n", numLookups );
    return false;
}

template <typename T>
__device__ bool hashProbeUnique1 ( unique_ht<T>* hash_table, int ht_size, uint64_t hash, int& numLookups, T** payload ) {
    int location;
    while ( numLookups < ht_size )  {
        location = ( hash + numLookups ) % ht_size;
        unique_ht<T>& elem = hash_table[location];
        numLookups++;
        if ( elem.hash == hash ) {
            *payload = &elem.payload;
            return true;
        } else if ( elem.hash == HASH_EMPTY ) {
            return false;
        }
    }
    //printf ( "probing full hash table - num lookups: %i\n", numLookups );
    return false;
}


// simplified version of multi ht without locking but with intermediate prefix sum
struct multi_ht {
    uint32_t offset;
    uint32_t count;
};


// intialize an array as used e.g. for join hash tables
__global__ void initMultiHT ( multi_ht* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].offset = 0xffffffff;
	ht[i].count = 0;
    }
}


// compact scan implementation to allocate entry buckets
__global__ void scanMultiHT ( multi_ht* ht, uint32_t num, int* range_offset ) {
    unsigned int mask = 0xffffffff;
    int warpSize = 32;
    int lane_id = threadIdx.x % warpSize;
    int lim = ( ( num + warpSize - 1 ) / warpSize ) * warpSize;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; i += blockDim.x * gridDim.x) {
        uint32_t val = 0;
        if ( i < num ) 
            val = ht[i].count;
        uint32_t scan = val;
        #pragma unroll
        for (int s = 1; s <= warpSize; s *= 2) {
            int n = __shfl_up_sync ( mask, scan, s, warpSize);
            if (lane_id >= s) scan += n;
        }
        uint32_t glob;
        if (threadIdx.x % warpSize == warpSize - 1) {
            glob = atomicAdd ( range_offset, scan );
        }
        glob = __shfl_sync ( mask, glob, warpSize -1, warpSize );
        if ( i < num ) {
            ht[i].offset = scan - val + glob;
            ht[i].count = 0;
	}
    }
}


// join hash insert: count the number of matching elements
__device__ void hashCountMulti ( multi_ht* ht, int32_t ht_size, uint64_t hash ) {
    int org_location = hash % ht_size;
    uint32_t location = org_location;
    multi_ht* entry = &ht[location];
    atomicAdd ( &(entry->count), 1);
}


// join hash insert: insert elements
template <typename T>
__device__ void hashInsertMulti ( multi_ht* ht, T* payload, int* range_offset, int32_t ht_size, uint64_t hash, T* payl ) {
    uint32_t location = hash % ht_size;
    multi_ht& entry = ht [ location ];
    uint32_t tupleOffset = atomicAdd ( &(entry.count), 1);
    payload [ tupleOffset  + entry.offset ] = *payl;
    return;
}


// join hash probe
__device__ bool hashProbeMulti ( multi_ht* ht, uint32_t ht_size, uint64_t hash, int& offset, int& end ) {
    uint32_t location = hash % ht_size;
    multi_ht& entry = ht [ location ];
    if ( entry.count != 0 ) {
        offset = entry.offset;
        end = offset + entry.count;
        return true;
    } else {
        return false;
    }
}


/*
 A lock that ensures that a section is only executed once.
 E.g. assigning the key to a ht entry
 */
 struct OnceLock {

    static const unsigned LOCK_FRESH   = 0;
    static const unsigned LOCK_WORKING = 1;
    static const unsigned LOCK_DONE    = 2;

    volatile unsigned lock;
    
    __device__ void init() {
        lock = LOCK_FRESH;
    } 

    __device__ bool enter() {
        unsigned lockState = atomicCAS ( (unsigned*) &lock, LOCK_FRESH, LOCK_WORKING );
        return lockState == LOCK_FRESH;
    }

    __device__ bool enter_w() {
        unsigned lockState = atomicCAS ( (unsigned*) &lock, LOCK_FRESH, LOCK_WORKING );
        return lock == LOCK_FRESH;
    }

    __device__ void done() {
        __threadfence();
        lock = LOCK_DONE;
        __threadfence();
    }

    __device__ void done_w() {
        lock = LOCK_DONE;
    }

    __device__ void wait() {
        while ( lock != LOCK_DONE );
    }
};


template <typename T>
struct __align__(8) agg_ht {
    OnceLock lock;
    uint64_t hash;
    T payload;
};

template <typename T>
__global__ void initAggHT ( agg_ht<T>* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].lock.init();
	ht[i].hash = HASH_EMPTY;
    }
}


// returns candidate bucket
template <typename T>
__device__ int hashAggregateGetBucket ( agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, T* payl ) {
    int location=-1;
    bool done=false;
    while ( !done ) {
        location = ( grouphash + numLookups ) % ht_size;
        agg_ht<T>& entry = ht [ location ];
        numLookups++;
        if ( entry.lock.enter() ) {
            entry.payload = *payl;
            entry.hash = grouphash;
            entry.lock.done();
        }
        entry.lock.wait();
        done = (entry.hash == grouphash);
        if ( numLookups == ht_size ) {
            printf ( "hash table full\n" );
            break;
	    }
    }
    return location;
}

template <typename T>
__device__ __forceinline__ int hashAggregateGetBucket ( agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, T* payl, int &doMemoryFence) {
    int location=-1;
    bool done=false;
    while ( !done ) {
        location = ( grouphash + numLookups ) % ht_size;
        agg_ht<T>& entry = ht [ location ];
        numLookups++;
        if ( entry.lock.enter() ) {
            entry.payload = *payl;
            entry.hash = grouphash;
            entry.lock.done();
            doMemoryFence = 1;
        } else {
            entry.lock.wait();
        } 
        
        done = (entry.hash == grouphash);
        if ( numLookups == ht_size ) {
            printf ( "hash table full\n" );
            break;
	    }
    }
    return location;
}


// return value indicates if more candidate buckets exist
// location used as return value for payload location
template <typename T>
__device__ bool hashAggregateFindBucket ( agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, int& location ) {
    location=-1;
    bool done=false;
    while ( !done ) {
        location = ( grouphash + numLookups++ ) % ht_size;
        if ( ht [ location ].hash == HASH_EMPTY ) {
            return false;
        }
        done = ( ht [ location ].hash == grouphash);
        
    }
    return true;
}

// returns candidate bucket
template <typename T>
__device__ __forceinline__ int hashAggregateGetBucketTest ( agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, T* payl ) {
    int location=-1;
    bool done=false;

    location =  ( grouphash + numLookups ) % ht_size;
    agg_ht<T>& entry = ht [ location ];
    
    unsigned lock_status = entry.lock.enter1();
    
    if (lock_status == OnceLock::LOCK_FRESH) {
        entry.payload = *payl;
        entry.hash = grouphash;
    }
    __syncwarp();
    
    if (entry.hash == grouphash) {

    } else if (lock_status == OnceLock::LOCK_DONE) {
        done = entry.hash == grouphash;
    }
    __threadfence();
    return location;
}

/*
#define CRYSTAL_HASH(X,Y,Z) ((X-Z) % Y)

template <typename T>
__device__ void hashBuildUniqueCrystal ( unique_ht<T>* hash_table, int ht_size, uint64_t hash, T* payload ) {
    int org_location = hash % ht_size;
    uint32_t location = org_location;
    unique_ht<T>* elem;

    while ( true )  {
        elem = &(hash_table[location]);
        unsigned long long probe_hash = atomicCAS( (unsigned long long*) &(elem->hash), HASH_EMPTY, (unsigned long long)hash );
        if(probe_hash == HASH_EMPTY) {
            elem->payload = *payload;
	    return;
        }
        location = (location + 1) % ht_size;
        if(location == org_location) {
            printf ( "build on full hash table, location: %i\n", location );
            return;
        }
    }
}

template <typename T>
__device__ bool hashProbeUniqueCrystal ( unique_ht<T>* hash_table, int ht_size, uint64_t hash, int& numLookups, int& location) { //,T** payload ) {
    //int location;
    while ( numLookups < ht_size )  {
        location = ( hash + numLookups ) % ht_size;
        unique_ht<T>& elem = hash_table[location];
        numLookups++;
        if ( elem.hash == hash ) {
            //*payload = &elem.payload;
            return true;
        } else if ( elem.hash == HASH_EMPTY ) {
            return false;
        }
    }
    //printf ( "probing full hash table - num lookups: %i\n", numLookups );
    return false;
}
*/

/*
template <typename T>
struct local_agg_ht {
    uint64_t hash;
    T payload;
}

template <typename T>
__device__ int hashAggregateGetBucket ( local_agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash ) {
    int loc = grouphash % ht_size;
    uint64_t h = ht[loc]
    return h == 0xFFFFFFFFFFFFFFFF || h == grouphash ? loc : -1;
}
*/
template <typename T>
__host__ __device__ void trieProbe (T* keys, int size, T &search_key, int &tid) {
    int start = tid + 1; 
    int end = size;
    while (start < end) {
        int tid = (start + end) / 2;
        bool a = search_key > keys[tid];
        start = a ? tid + 1 : start;
        end = a ? end : tid;
    }
}

__host__ __device__ __forceinline__ int trieSize(int* offset, int tid) {
    return offset[tid*2+1] - offset[tid*2];
}

__host__ __device__ __forceinline__ bool trieSearch(int* keys, int* offset, int search_key, int tid, int& new_tid) {
    int start = offset[tid*2];
    int end = offset[tid*2+1];
    new_tid = -1;
    //if (threadIdx.x == 0) printf("thread_id: %d, start: %d, end: %d, search_key: %d\n", threadIdx.x, start, end, search_key);
    if (tid == -1) return false;
    while (start < end) {
        int t = (start + end) / 2;
        int k = keys[t];
        //if (threadIdx.x == 0) printf("t: %d, start: %d, end: %d, search_key: %d, k: %d\n", t, start, end, search_key, k);
        if (k < search_key) {
            start = t+1;
        } else if (k > search_key) {
            end = t;
        } else {
            new_tid = t;
            //if (threadIdx.x == 0) printf("found start: %d, end: %d, search_key: %d\n", start, end, search_key);
            return true;
        } 
    }
    //if (threadIdx.x == 0) printf("not found thread_id: %d, start: %d, end: %d, search_key: %d\n", threadIdx.x, start, end, search_key);
    return false;
}

__device__ __forceinline__ bool binarySearch(int* array, int start, int end, int key) {

    while (start < end) {
        int t = (start + end) / 2;
        int k = array[t];
        if (k < key) {
            start = t+1;
        } else if (k > key) {
            end = t;
        } else {
            return true;
        }
    }
    return false;
}


// a1 == 1; // attributes..

// a2 == 2; // attributes...

// a3 == 3; // attr




__device__ void csrProbeMulti ( int* offsets, int hash, int& offset, int& end ) {
    offset = hash == 0 ? 0 : offsets[hash-1];
    end = offsets[hash];
}


__forceinline__ __device__ bool indexProbeMulti ( const int* offsets, int tid, int& offset, int& end ) {
    offset = offsets[2*tid];
    end = offsets[2*tid+1];
    return offset < end;
}

__forceinline__ __device__ int indexGetPid ( const int* t, int lid ) {
    return t[lid];
    //return __ldg(t + lid);
}

__forceinline__ __device__ void sample(int gpart_id, int thread_id, unsigned long long* samples, unsigned long long start, unsigned long long type) {
    if (thread_id == 0) {
        ++samples[0];
        assert(samples[0] < (1 << 17));
        /*
        if (samples[0] > (1 << 17)) {
            printf("%lld\n", samples[0]);
        }
        assert(samples[0] < (1 << 17));
        */
        //uint32_t smid32;    
        //asm volatile("mov.u32 %0, %%smid;" : "=r"(smid32));
        //uint64_t smid = (uint64_t) smid32;
        unsigned long long p = clock64();
        uint64_t smid = gpart_id != -1 ? gpart_id : 32; //0;
        start = 0;
        samples[samples[0]] = (p > start ? p - start : 0) | (type << 60) | (smid << 52);
    }
}

__global__ void krnl_sample_start(unsigned long long* sample_start) {
    if (threadIdx.x == 0) sample_start[0] = clock64();
}

#endif