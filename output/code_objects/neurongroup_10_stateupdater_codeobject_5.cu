#include "code_objects/neurongroup_10_stateupdater_codeobject_5.h"
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
        
    template <typename T>
    __host__ __device__
    double _brian_exp(T value)
    {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return exp((double)value);
    #else
        return exp(value);
    #endif
    }
    inline __host__ __device__
    float _brian_exp(float value)
    {
        return exp(value);
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
_run_kernel_neurongroup_10_stateupdater_codeobject_5(
    int _N,
    int THREADS_PER_BLOCK,
    ///// KERNEL_PARAMETERS /////
    double* _ptr_array_neurongroup_10_A,
	double* _ptr_array_neurongroup_10_Iexc,
	double* _ptr_array_neurongroup_10_Iinh,
	double* _ptr_array_neurongroup_10_X,
	double* _ptr_array_neurongroup_10_Y,
	const double _value_array_defaultclock_dt,
	double* _ptr_array_neurongroup_10_v
    )
{
    using namespace brian;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int _idx = bid * blockDim.x + tid;
    int _vectorisation_idx = _idx;

    ///// KERNEL_CONSTANTS /////
    const int _numA = 16;
	const double Cm = 2.81e-10;
	const double DeltaT = 0.002;
	const double EL = - 0.0706;
	const int _numIexc = 16;
	const int _numIinh = 16;
	const int64_t N = 16;
	const double Vth = - 0.0504;
	const int _numX = 16;
	const int _numY = 16;
	const double c = 4e-09;
	const double gL = 3.0000000000000004e-08;
	const double tau_A = 0.001;
	const double tau_decay = 0.05;
	const double tau_rise = 0.005;
	const int _numv = 16;

    ///// kernel_lines /////
        
    const double* _ptr_array_defaultclock_dt = &_value_array_defaultclock_dt;



    if(_vectorisation_idx >= _N)
    {
        return;
    }



    ///// scalar_code /////
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double _lio_1 = 1.0f*dt/tau_A;
    const double _lio_2 = - EL;
    const double _lio_3 = 1.0f*1.0/tau_decay;
    const double _lio_4 = 1.0f*1.0/tau_rise;
    const double _lio_5 = 1.0f*dt/tau_decay;
    const double _lio_6 = 1.0f*dt/Cm;
    const double _lio_7 = DeltaT * gL;
    const double _lio_8 = 1.0f*1.0/DeltaT;
    const double _lio_9 = - Vth;


    {
        ///// vector_code /////
                
        double A = _ptr_array_neurongroup_10_A[_idx];
        const double Iexc = _ptr_array_neurongroup_10_Iexc[_idx];
        const double Iinh = _ptr_array_neurongroup_10_Iinh[_idx];
        double X = _ptr_array_neurongroup_10_X[_idx];
        double Y = _ptr_array_neurongroup_10_Y[_idx];
        const double dt = _ptr_array_defaultclock_dt[0];
        double v = _ptr_array_neurongroup_10_v[_idx];
        const double _A = A + (_lio_1 * ((- A) + (c * (_lio_2 + v))));
        const double _X = X + (dt * ((_lio_3 * (- X)) + (_lio_4 * Y)));
        const double _Y = (_lio_5 * (- Y)) + Y;
        const double _v = v + (_lio_6 * (((((- A) + (_lio_7 * _brian_exp(_lio_8 * (_lio_9 + v)))) + Iexc) + Iinh) - (gL * (_lio_2 + v))));
        A = _A;
        X = _X;
        Y = _Y;
        v = _v;
        _ptr_array_neurongroup_10_A[_idx] = A;
        _ptr_array_neurongroup_10_X[_idx] = X;
        _ptr_array_neurongroup_10_Y[_idx] = Y;
        _ptr_array_neurongroup_10_v[_idx] = v;


    }
}



void _run_neurongroup_10_stateupdater_codeobject_5()
{
    using namespace brian;


    ///// HOST_CONSTANTS ///////////
    const int _numA = 16;
		const double Cm = 2.81e-10;
		const double DeltaT = 0.002;
		const double EL = - 0.0706;
		const int _numIexc = 16;
		const int _numIinh = 16;
		const int64_t N = 16;
		const double Vth = - 0.0504;
		const int _numX = 16;
		const int _numY = 16;
		const double c = 4e-09;
		const double gL = 3.0000000000000004e-08;
		const double tau_A = 0.001;
		const double tau_decay = 0.05;
		const double tau_rise = 0.005;
		const int _numv = 16;

    const int _N = N;

    ///// ADDITIONAL_HOST_CODE /////
    


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
                    _run_kernel_neurongroup_10_stateupdater_codeobject_5, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
                );

        // Round up according to array size
        num_blocks = (_N + num_threads - 1) / num_threads;





        // calculate theoretical occupancy
        int max_active_blocks;
        CUDA_SAFE_CALL(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                    _run_kernel_neurongroup_10_stateupdater_codeobject_5, num_threads, needed_shared_memory)
                );

        float occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                          (float)(max_threads_per_sm / num_threads_per_warp);


        // check if we have enough ressources to call kernel with given number
        // of blocks and threads (can only occur for the else case above as for the
        // first max. occupancy)
        struct cudaFuncAttributes funcAttrib;
        CUDA_SAFE_CALL(
                cudaFuncGetAttributes(&funcAttrib, _run_kernel_neurongroup_10_stateupdater_codeobject_5)
                );
        if (num_threads > funcAttrib.maxThreadsPerBlock)
        {
            // use the max num_threads before launch failure
            num_threads = funcAttrib.maxThreadsPerBlock;
            printf("WARNING Not enough ressources available to call "
                   "_run_kernel_neurongroup_10_stateupdater_codeobject_5 "
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
                        _run_kernel_neurongroup_10_stateupdater_codeobject_5, num_threads, needed_shared_memory)
                    );

            occupancy = (max_active_blocks * num_threads / num_threads_per_warp) /
                        (float)(max_threads_per_sm / num_threads_per_warp);
        }


        else
        {
            printf("INFO _run_kernel_neurongroup_10_stateupdater_codeobject_5\n"
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


    _run_kernel_neurongroup_10_stateupdater_codeobject_5<<<num_blocks, num_threads>>>(
            _N,
            num_threads,
            ///// HOST_PARAMETERS /////
            dev_array_neurongroup_10_A,
			dev_array_neurongroup_10_Iexc,
			dev_array_neurongroup_10_Iinh,
			dev_array_neurongroup_10_X,
			dev_array_neurongroup_10_Y,
			_array_defaultclock_dt[0],
			dev_array_neurongroup_10_v
        );

    CUDA_CHECK_ERROR("_run_kernel_neurongroup_10_stateupdater_codeobject_5");


}


