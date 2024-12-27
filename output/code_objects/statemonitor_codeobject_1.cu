#include "code_objects/statemonitor_codeobject_1.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>



////// SUPPORT CODE ///////
namespace {
    randomNumber_t _host_rand(const int _vectorisation_idx);
    randomNumber_t _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

    ///// support_code_lines /////
        
                        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                        __device__ double* _namespace_timedarray_3_values;
                        #else
                        double* _namespace_timedarray_3_values;
                        #endif
    __host__ __device__
    static inline double _timedarray_3(const double t, const int i)
    {
        const double epsilon = 0.100000000000000006 / 8192;
        if (i < 0 || i >= 225)
            return NAN;
        int timestep = (int)((t/epsilon + 0.5)/8192);
        if(timestep < 0)
           timestep = 0;
        else if(timestep >= 40)
            timestep = 40-1;
        return _namespace_timedarray_3_values[timestep*225 + i];
    }
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    __host__ __device__ static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif
                    inline __device__ int _brian_atomicAdd(int* address, int val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ float _brian_atomicAdd(float* address, float val)
                    {
            // hardware implementation
            return atomicAdd(address, val);
                    }
                    inline __device__ double _brian_atomicAdd(double* address, double val)
                    {
                            #if (__CUDA_ARCH__ >= 600)
            // hardware implementation
            return atomicAdd(address, val);
                            #else
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val +
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                            #endif
                    }
                    inline __device__ int _brian_atomicMul(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val * assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicMul(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val *
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicMul(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val *
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }
                    inline __device__ int _brian_atomicDiv(int* address, int val)
                    {
                        // software implementation
                        int old = *address, assumed;
                        do {
                            assumed = old;
                            old = atomicCAS(address, assumed, val / assumed);
                        } while (assumed != old);
                        return old;
                    }
                    inline __device__ float _brian_atomicDiv(float* address, float val)
                    {
            // software implementation
            int* address_as_int = (int*)address;
            int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(val /
                                       __int_as_float(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __int_as_float(old);
                    }
                    inline __device__ double _brian_atomicDiv(double* address, double val)
                    {
            // software implementation
            unsigned long long int* address_as_int = (unsigned long long int*)address;
            unsigned long long int old = *address_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __double_as_longlong(val /
                                       __longlong_as_double(assumed)));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);
            return __longlong_as_double(old);
                    }


    // Implement dummy functions such that the host compiled code of binomial
    // functions works. Hacky, hacky ...
    randomNumber_t _host_rand(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    randomNumber_t _host_randn(const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_rand` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx)
    {
        printf("ERROR: Called dummy function `_host_poisson` in %s:%d\n", __FILE__,
                __LINE__);
        exit(EXIT_FAILURE);
    }
}

////// hashdefine_lines ///////



__global__ void
_run_kernel_statemonitor_codeobject_1(
    int _N,
    int THREADS_PER_BLOCK,
    int current_iteration,
    double** monitor_I,
    double** monitor_v,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_statemonitor_N,
	int32_t* _ptr_array_neurongroup_indices,
	const double _value_array_defaultclock_t,
	int32_t* _ptr_array_statemonitor__indices,
	double* _ptr_array_neurongroup_v,
	double* _ptr_array_statemonitor_t,
	const int _numt
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numN = 1;
	const int _num__source_I_neurongroup_indices = 2;
	const int _num_indices = 1;
	const int _num_source_v = 2;

    ///// kernel_lines /////
        
                        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                        _namespace_timedarray_3_values = d_timedarray_3_values;
                        #else
                        _namespace_timedarray_3_values = _timedarray_3_values;
                        #endif
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;



    if(_vectorisation_idx >= _N)
    {
        return;
    }

    _idx = _ptr_array_statemonitor__indices[_vectorisation_idx];


    ///// scalar_code /////
        


    {
        ///// vector_code /////
                
        const int32_t __source_I_neurongroup_indices = _ptr_array_neurongroup_indices[_idx];
        const double __source_I_neurongroup_t = _ptr_array_defaultclock_t[0];
        const double _source_v = _ptr_array_neurongroup_v[_idx];
        const double _to_record_v = _source_v;
        const double _source_I = _timedarray_3(__source_I_neurongroup_t, __source_I_neurongroup_indices);
        const double _to_record_I = _source_I;


    monitor_I[_vectorisation_idx][current_iteration] = _to_record_I;
    monitor_v[_vectorisation_idx][current_iteration] = _to_record_v;
    }
}



void _run_statemonitor_codeobject_1()
{
    using namespace brian;


    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		const int _num__source_I_neurongroup_indices = 2;
		const int _num_indices = 1;
		const int _num_source_v = 2;
		double* const dev_array_statemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_statemonitor_t[0]);
		const int _numt = dev_dynamic_array_statemonitor_t.size();


    ///// ADDITIONAL_HOST_CODE /////
    

// NOTE: We are using _N as the number of recorded indices here (the relevant size for
// parallelization). This is different from `StateMonitor.N` in Python, which refers to
// the number of recorded time steps (while `StateMonitor.n_indices` gives the number of
// recorded indices).
const int _N = _num_indices;

// We are using an extra variable because HOST_CONSTANTS uses the device vector, which
// is not used (TODO: Fix this in HOST_CONSTANTS instead of this hack here...)
const int _numt_host = _dynamic_array_statemonitor_t.size();

// We push t only on host and don't make a device->host copy in write_arrays()
_dynamic_array_statemonitor_t.push_back(defaultclock.t[0]);

// Update size variables for Python side indexing to work
_array_statemonitor_N[0] += 1;

int num_iterations = defaultclock.i_end;
int current_iteration = defaultclock.timestep[0];
static int start_offset = current_iteration - _numt_host;

    static int num_threads, num_blocks;
    static size_t needed_shared_memory = 0;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    _run_kernel_statemonitor_codeobject_1, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;


addresses_monitor__dynamic_array_statemonitor_I.clear();
addresses_monitor__dynamic_array_statemonitor_v.clear();
for(int i = 0; i < _num__array_statemonitor__indices; i++)
{
    _dynamic_array_statemonitor_I[i].resize(_numt_host + num_iterations - current_iteration);
    addresses_monitor__dynamic_array_statemonitor_I.push_back(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_I[i][0]));
    _dynamic_array_statemonitor_v[i].resize(_numt_host + num_iterations - current_iteration);
    addresses_monitor__dynamic_array_statemonitor_v.push_back(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_v[i][0]));
}



        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_statemonitor_codeobject_1, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_statemonitor_codeobject_1)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_statemonitor_codeobject_1 "
                   "with maximum possible threads per block (%u). "
                   "Reducing num_threads to %u. (Kernel needs %i "
                   "registers per block, %i bytes of "
                   "statically-allocated shared memory per block, %i "
                   "bytes of local memory per thread and a total of %i "
                   "bytes of user-allocated constant memory)\n",
                   max_threads_per_block, num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes);

            // calculate theoretical occupancy for new num_threads
            CUDA_SAFE_CALL(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                        _run_kernel_statemonitor_codeobject_1, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_statemonitor_codeobject_1\n"
                   "\t%u blocks\n"
                   "\t%u threads\n"
                   "\t%i registers per thread\n"
                   "\t%i bytes statically-allocated shared memory per block\n"
                   "\t%i bytes local memory per thread\n"
                   "\t%i bytes user-allocated constant memory\n"
                   "\t%.3f theoretical occupancy\n",
                   num_blocks,
                   num_threads, funcAttrib.numRegs,
                   funcAttrib.sharedSizeBytes, funcAttrib.localSizeBytes,
                   funcAttrib.constSizeBytes, occupancy);
        }
        first_run = false;
    }

// If the StateMonitor is run outside the MagicNetwork, we need to resize it.
// Happens e.g. when StateMonitor.record_single_timestep() is called.
if(current_iteration >= num_iterations)
{
    for(int i = 0; i < _num__array_statemonitor__indices; i++)
    {
        _dynamic_array_statemonitor_I[i].resize(_numt_host + 1);
        addresses_monitor__dynamic_array_statemonitor_I[i] = thrust::raw_pointer_cast(&_dynamic_array_statemonitor_I[i][0]);
        _dynamic_array_statemonitor_v[i].resize(_numt_host + 1);
        addresses_monitor__dynamic_array_statemonitor_v[i] = thrust::raw_pointer_cast(&_dynamic_array_statemonitor_v[i][0]);
    }
}

// TODO we get invalid launch configuration if this is 0, which happens e.g. for StateMonitor(..., variables=[])
if (_num__array_statemonitor__indices > 0)
{

    _run_kernel_statemonitor_codeobject_1<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
    current_iteration - start_offset,
    thrust::raw_pointer_cast(&addresses_monitor__dynamic_array_statemonitor_I[0]),
    thrust::raw_pointer_cast(&addresses_monitor__dynamic_array_statemonitor_v[0]),
            ///// HOST_PARAMETERS /////
            dev_array_statemonitor_N,
			dev_array_neurongroup_indices,
			_array_defaultclock_t[0],
			dev_array_statemonitor__indices,
			dev_array_neurongroup_v,
			dev_array_statemonitor_t,
			_numt
        );

    CUDA_CHECK_ERROR("_run_kernel_statemonitor_codeobject_1");

}

}


