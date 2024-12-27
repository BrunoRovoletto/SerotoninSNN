#include "code_objects/spikemonitor_codeobject_1.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


////// SUPPORT CODE ///////
namespace {
    randomNumber_t _host_rand(const int _vectorisation_idx);
    randomNumber_t _host_randn(const int _vectorisation_idx);
    int32_t _host_poisson(double _lambda, const int _vectorisation_idx);

    ///// block extra_device_helper /////

    ///// support_code_lines /////
        
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
_run_kernel_spikemonitor_codeobject_1(
    int _N,
    int THREADS_PER_BLOCK,
        int _monitor_size,
        int32_t* _eventspace,
        double* _new_ptr_array_spikemonitor_t,
        int32_t* _new_ptr_array_spikemonitor_i,
    ///// KERNEL_PARAMETERS /////
    int32_t* _ptr_array_spikemonitor_N,
	int32_t* _ptr_array_neurongroup_i,
	int32_t* _ptr_array_spikemonitor__source_idx,
	const double _value_array_defaultclock_t,
	int32_t* _ptr_array_neurongroup__spikespace,
	int32_t* _ptr_array_spikemonitor_count,
	int32_t* _ptr_array_spikemonitor_i,
	const int _numi,
	double* _ptr_array_spikemonitor_t,
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
	const int _num_source_i = 2;
	const int _num_source_idx = 2;
	const int64_t _source_start = 0;
	const int64_t _source_stop = 2;
	const int _num_spikespace = 3;
	const int _numcount = 2;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_t = &_value_array_defaultclock_t;



    if(_vectorisation_idx >= _N)
    {
        return;
    }



// Eventspace is filled from left with all neuron IDs that triggered an event, rest -1
int32_t spiking_neuron = _eventspace[_idx];
assert(spiking_neuron != -1);

int _monitor_idx = _vectorisation_idx + _monitor_size;
_idx = spiking_neuron;
_vectorisation_idx = _idx;

// vector_code

const int32_t _source_i = _ptr_array_neurongroup_i[_idx];
const double _source_t = _ptr_array_defaultclock_t[0];
const int32_t _to_record_i = _source_i;
const double _to_record_t = _source_t;



// fill the monitors
_new_ptr_array_spikemonitor_t[_monitor_idx] = _to_record_t;
_new_ptr_array_spikemonitor_i[_monitor_idx] = _to_record_i;

_ptr_array_spikemonitor_count[_idx - _source_start]++;
}



void _run_spikemonitor_codeobject_1()
{
    using namespace brian;


    ///// HOST_CONSTANTS ///////////
    const int _numN = 1;
		const int _num_source_i = 2;
		const int _num_source_idx = 2;
		const int64_t _source_start = 0;
		const int64_t _source_stop = 2;
		const int _num_spikespace = 3;
		const int _numcount = 2;
		int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_i.size();
		double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_t.size();

// The _N of this kernel (total number of threads) is defined by the number of events
int _N;

    ///// ADDITIONAL_HOST_CODE /////
    


static int num_threads;
int num_events, num_blocks;
    static size_t needed_shared_memory = 0;
    static bool first_run = true;
    if (first_run)
    {
        // get number of blocks and threads
        int min_num_threads; // The minimum grid size needed to achieve the
                             // maximum occupancy for a full device launch

        CUDA_SAFE_CALL(
                cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
                    _run_kernel_spikemonitor_codeobject_1, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;





        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_spikemonitor_codeobject_1, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_spikemonitor_codeobject_1)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_spikemonitor_codeobject_1 "
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
                        _run_kernel_spikemonitor_codeobject_1, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_spikemonitor_codeobject_1\n"
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



// Number of events in eventspace
int _num_events, _num_events_subgroup;
int32_t* _eventspace = dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace];
CUDA_SAFE_CALL(
        cudaMemcpy(
            &_num_events,
            &_eventspace[_num_spikespace - 1],
            sizeof(int32_t),
            cudaMemcpyDeviceToHost
            )
        );

// Get the number of events
_N = _num_events;

// Get current size of device vectors
int _monitor_size = dev_dynamic_array_spikemonitor_t.size();

// Increase device vectors based on number of events
THRUST_CHECK_ERROR(
        dev_dynamic_array_spikemonitor_t.resize(_monitor_size + _N)
        );
THRUST_CHECK_ERROR(
        dev_dynamic_array_spikemonitor_i.resize(_monitor_size + _N)
        );

// Round up number of blocks according to number of events
num_blocks = (_N + num_threads - 1) / num_threads;

// Only call kernel if there are events in the current time step
if (_N > 0)
{
    _run_kernel_spikemonitor_codeobject_1<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            _monitor_size,
            _eventspace,
            thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]),
            ///// HOST_PARAMETERS /////
            dev_array_spikemonitor_N,
			dev_array_neurongroup_i,
			dev_array_spikemonitor__source_idx,
			_array_defaultclock_t[0],
			dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
			dev_array_spikemonitor_count,
			dev_array_spikemonitor_i,
			_numi,
			dev_array_spikemonitor_t,
			_numt
        );

    CUDA_CHECK_ERROR("_run_kernel_spikemonitor_codeobject_1");
}

// Increase total number of events in monitor
_array_spikemonitor_N[0] += _N;


}

void _debugmsg_spikemonitor_codeobject_1()
{
    using namespace brian;

    // HOST_CONSTANTS
    const int _numN = 1;
		const int _num_source_i = 2;
		const int _num_source_idx = 2;
		const int64_t _source_start = 0;
		const int64_t _source_stop = 2;
		const int _num_spikespace = 3;
		const int _numcount = 2;
		int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_i[0]);
		const int _numi = dev_dynamic_array_spikemonitor_i.size();
		double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&dev_dynamic_array_spikemonitor_t[0]);
		const int _numt = dev_dynamic_array_spikemonitor_t.size();

    printf("Number of spikes: %d\n", _array_spikemonitor_N[0]);
}

