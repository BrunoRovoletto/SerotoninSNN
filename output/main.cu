#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "rand.h"

#include "code_objects/synapses_synapses_create_array_codeobject.h"
#include "code_objects/synapses_synapses_create_array_codeobject_1.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_3_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_4_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_4_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_5_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_7_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_7_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_8_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_8_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_9_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_9_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_10_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_10_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_11_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_11_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_12_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_12_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_13_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_13_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_14_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_14_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_15_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_15_group_variable_set_conditional_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_10_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_11_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_12_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_13_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_14_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_15_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_1_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_2_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_3_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_4_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_5_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_6_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_7_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_8_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_9_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_summed_variable_Iexc2_post_codeobject.h"
#include "code_objects/neurongroup_10_stateupdater_codeobject.h"
#include "code_objects/neurongroup_11_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/neurongroup_3_stateupdater_codeobject.h"
#include "code_objects/neurongroup_4_stateupdater_codeobject.h"
#include "code_objects/neurongroup_5_stateupdater_codeobject.h"
#include "code_objects/neurongroup_6_stateupdater_codeobject.h"
#include "code_objects/neurongroup_7_stateupdater_codeobject.h"
#include "code_objects/neurongroup_8_stateupdater_codeobject.h"
#include "code_objects/neurongroup_9_stateupdater_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_10_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_10_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_11_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_11_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_3_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_3_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_4_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_4_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_5_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_5_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_6_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_6_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_7_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_7_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_8_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_8_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_9_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_9_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/synapses_10_pre_push_spikes.h"
#include "code_objects/before_run_synapses_10_pre_push_spikes.h"
#include "code_objects/synapses_10_pre_codeobject.h"
#include "code_objects/synapses_11_pre_push_spikes.h"
#include "code_objects/before_run_synapses_11_pre_push_spikes.h"
#include "code_objects/synapses_11_pre_codeobject.h"
#include "code_objects/synapses_12_pre_push_spikes.h"
#include "code_objects/before_run_synapses_12_pre_push_spikes.h"
#include "code_objects/synapses_12_pre_codeobject.h"
#include "code_objects/synapses_13_pre_push_spikes.h"
#include "code_objects/before_run_synapses_13_pre_push_spikes.h"
#include "code_objects/synapses_13_pre_codeobject.h"
#include "code_objects/synapses_14_pre_push_spikes.h"
#include "code_objects/before_run_synapses_14_pre_push_spikes.h"
#include "code_objects/synapses_14_pre_codeobject.h"
#include "code_objects/synapses_15_pre_push_spikes.h"
#include "code_objects/before_run_synapses_15_pre_push_spikes.h"
#include "code_objects/synapses_15_pre_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/before_run_synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/before_run_synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/before_run_synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/before_run_synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_7_pre_push_spikes.h"
#include "code_objects/before_run_synapses_7_pre_push_spikes.h"
#include "code_objects/synapses_7_pre_codeobject.h"
#include "code_objects/synapses_8_pre_push_spikes.h"
#include "code_objects/before_run_synapses_8_pre_push_spikes.h"
#include "code_objects/synapses_8_pre_codeobject.h"
#include "code_objects/synapses_9_pre_push_spikes.h"
#include "code_objects/before_run_synapses_9_pre_push_spikes.h"
#include "code_objects/synapses_9_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/neurongroup_10_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_11_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_2_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_3_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_4_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_5_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_6_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_7_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_8_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_9_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>
#include "cuda_profiler_api.h"




void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() >=2 && args[0] == "--results_dir")
    {
        brian::results_dir = args[1];
        #ifdef DEBUG
        std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
        #endif
        args.erase(args.begin(), args.begin()+2);
    }
        


    // seed variable set in Python through brian2.seed() calls can use this
    // variable (see device.py CUDAStandaloneDevice.generate_main_source())
    unsigned long long seed;

    //const std::clock_t _start_time = std::clock();

    CUDA_SAFE_CALL(
            cudaSetDevice(0)
            );

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );
    size_t limit = 128 * 1024 * 1024;
    CUDA_SAFE_CALL(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit)
            );
    CUDA_SAFE_CALL(
            cudaDeviceSynchronize()
            );

    //const double _run_time2 = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: setting cudaDevice stuff took %f seconds\n", _run_time2);

    brian_start();

        


    //const std::clock_t _start_time3 = std::clock();
    {
        using namespace brian;

                
        for(int i=0; i<_num__array_neurongroup__spikespace; i++)
        {
            _array_neurongroup__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace],
                &_array_neurongroup__spikespace[0],
                sizeof(_array_neurongroup__spikespace[0])*_num__array_neurongroup__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup__spikespace[current_idx_array_neurongroup__spikespace] + _num__array_neurongroup__spikespace - 1,
                                    &_array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1],
                                    sizeof(_array_neurongroup__spikespace[_num__array_neurongroup__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_9__spikespace; i++)
        {
            _array_neurongroup_9__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_9__spikespace[current_idx_array_neurongroup_9__spikespace],
                &_array_neurongroup_9__spikespace[0],
                sizeof(_array_neurongroup_9__spikespace[0])*_num__array_neurongroup_9__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_9__spikespace[_num__array_neurongroup_9__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_9__spikespace[current_idx_array_neurongroup_9__spikespace] + _num__array_neurongroup_9__spikespace - 1,
                                    &_array_neurongroup_9__spikespace[_num__array_neurongroup_9__spikespace - 1],
                                    sizeof(_array_neurongroup_9__spikespace[_num__array_neurongroup_9__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_8__spikespace; i++)
        {
            _array_neurongroup_8__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_8__spikespace[current_idx_array_neurongroup_8__spikespace],
                &_array_neurongroup_8__spikespace[0],
                sizeof(_array_neurongroup_8__spikespace[0])*_num__array_neurongroup_8__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_8__spikespace[_num__array_neurongroup_8__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_8__spikespace[current_idx_array_neurongroup_8__spikespace] + _num__array_neurongroup_8__spikespace - 1,
                                    &_array_neurongroup_8__spikespace[_num__array_neurongroup_8__spikespace - 1],
                                    sizeof(_array_neurongroup_8__spikespace[_num__array_neurongroup_8__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_7__spikespace; i++)
        {
            _array_neurongroup_7__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_7__spikespace[current_idx_array_neurongroup_7__spikespace],
                &_array_neurongroup_7__spikespace[0],
                sizeof(_array_neurongroup_7__spikespace[0])*_num__array_neurongroup_7__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_7__spikespace[_num__array_neurongroup_7__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_7__spikespace[current_idx_array_neurongroup_7__spikespace] + _num__array_neurongroup_7__spikespace - 1,
                                    &_array_neurongroup_7__spikespace[_num__array_neurongroup_7__spikespace - 1],
                                    sizeof(_array_neurongroup_7__spikespace[_num__array_neurongroup_7__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_6__spikespace; i++)
        {
            _array_neurongroup_6__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_6__spikespace[current_idx_array_neurongroup_6__spikespace],
                &_array_neurongroup_6__spikespace[0],
                sizeof(_array_neurongroup_6__spikespace[0])*_num__array_neurongroup_6__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_6__spikespace[_num__array_neurongroup_6__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_6__spikespace[current_idx_array_neurongroup_6__spikespace] + _num__array_neurongroup_6__spikespace - 1,
                                    &_array_neurongroup_6__spikespace[_num__array_neurongroup_6__spikespace - 1],
                                    sizeof(_array_neurongroup_6__spikespace[_num__array_neurongroup_6__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_5__spikespace; i++)
        {
            _array_neurongroup_5__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_5__spikespace[current_idx_array_neurongroup_5__spikespace],
                &_array_neurongroup_5__spikespace[0],
                sizeof(_array_neurongroup_5__spikespace[0])*_num__array_neurongroup_5__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_5__spikespace[_num__array_neurongroup_5__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_5__spikespace[current_idx_array_neurongroup_5__spikespace] + _num__array_neurongroup_5__spikespace - 1,
                                    &_array_neurongroup_5__spikespace[_num__array_neurongroup_5__spikespace - 1],
                                    sizeof(_array_neurongroup_5__spikespace[_num__array_neurongroup_5__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_4__spikespace; i++)
        {
            _array_neurongroup_4__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_4__spikespace[current_idx_array_neurongroup_4__spikespace],
                &_array_neurongroup_4__spikespace[0],
                sizeof(_array_neurongroup_4__spikespace[0])*_num__array_neurongroup_4__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_4__spikespace[_num__array_neurongroup_4__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_4__spikespace[current_idx_array_neurongroup_4__spikespace] + _num__array_neurongroup_4__spikespace - 1,
                                    &_array_neurongroup_4__spikespace[_num__array_neurongroup_4__spikespace - 1],
                                    sizeof(_array_neurongroup_4__spikespace[_num__array_neurongroup_4__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_3__spikespace; i++)
        {
            _array_neurongroup_3__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_3__spikespace[current_idx_array_neurongroup_3__spikespace],
                &_array_neurongroup_3__spikespace[0],
                sizeof(_array_neurongroup_3__spikespace[0])*_num__array_neurongroup_3__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_3__spikespace[_num__array_neurongroup_3__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_3__spikespace[current_idx_array_neurongroup_3__spikespace] + _num__array_neurongroup_3__spikespace - 1,
                                    &_array_neurongroup_3__spikespace[_num__array_neurongroup_3__spikespace - 1],
                                    sizeof(_array_neurongroup_3__spikespace[_num__array_neurongroup_3__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_2__spikespace; i++)
        {
            _array_neurongroup_2__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace],
                &_array_neurongroup_2__spikespace[0],
                sizeof(_array_neurongroup_2__spikespace[0])*_num__array_neurongroup_2__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_2__spikespace[current_idx_array_neurongroup_2__spikespace] + _num__array_neurongroup_2__spikespace - 1,
                                    &_array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1],
                                    sizeof(_array_neurongroup_2__spikespace[_num__array_neurongroup_2__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_1__spikespace; i++)
        {
            _array_neurongroup_1__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace],
                &_array_neurongroup_1__spikespace[0],
                sizeof(_array_neurongroup_1__spikespace[0])*_num__array_neurongroup_1__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_1__spikespace[current_idx_array_neurongroup_1__spikespace] + _num__array_neurongroup_1__spikespace - 1,
                                    &_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1],
                                    sizeof(_array_neurongroup_1__spikespace[_num__array_neurongroup_1__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_11__spikespace; i++)
        {
            _array_neurongroup_11__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_11__spikespace[current_idx_array_neurongroup_11__spikespace],
                &_array_neurongroup_11__spikespace[0],
                sizeof(_array_neurongroup_11__spikespace[0])*_num__array_neurongroup_11__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_11__spikespace[_num__array_neurongroup_11__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_11__spikespace[current_idx_array_neurongroup_11__spikespace] + _num__array_neurongroup_11__spikespace - 1,
                                    &_array_neurongroup_11__spikespace[_num__array_neurongroup_11__spikespace - 1],
                                    sizeof(_array_neurongroup_11__spikespace[_num__array_neurongroup_11__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        for(int i=0; i<_num__array_neurongroup_10__spikespace; i++)
        {
            _array_neurongroup_10__spikespace[i] = - 1;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_10__spikespace[current_idx_array_neurongroup_10__spikespace],
                &_array_neurongroup_10__spikespace[0],
                sizeof(_array_neurongroup_10__spikespace[0])*_num__array_neurongroup_10__spikespace,
                cudaMemcpyHostToDevice
            )
        );
        _array_neurongroup_10__spikespace[_num__array_neurongroup_10__spikespace - 1] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_neurongroup_10__spikespace[current_idx_array_neurongroup_10__spikespace] + _num__array_neurongroup_10__spikespace - 1,
                                    &_array_neurongroup_10__spikespace[_num__array_neurongroup_10__spikespace - 1],
                                    sizeof(_array_neurongroup_10__spikespace[_num__array_neurongroup_10__spikespace - 1]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        for(int i=0; i<_num__array_neurongroup_v; i++)
        {
            _array_neurongroup_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_v,
                &_array_neurongroup_v[0],
                sizeof(_array_neurongroup_v[0])*_num__array_neurongroup_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_1_v; i++)
        {
            _array_neurongroup_1_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_1_v,
                &_array_neurongroup_1_v[0],
                sizeof(_array_neurongroup_1_v[0])*_num__array_neurongroup_1_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_2_v; i++)
        {
            _array_neurongroup_2_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_2_v,
                &_array_neurongroup_2_v[0],
                sizeof(_array_neurongroup_2_v[0])*_num__array_neurongroup_2_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_3_v; i++)
        {
            _array_neurongroup_3_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_3_v,
                &_array_neurongroup_3_v[0],
                sizeof(_array_neurongroup_3_v[0])*_num__array_neurongroup_3_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_4_v; i++)
        {
            _array_neurongroup_4_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_4_v,
                &_array_neurongroup_4_v[0],
                sizeof(_array_neurongroup_4_v[0])*_num__array_neurongroup_4_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_5_v; i++)
        {
            _array_neurongroup_5_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_5_v,
                &_array_neurongroup_5_v[0],
                sizeof(_array_neurongroup_5_v[0])*_num__array_neurongroup_5_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_6_v; i++)
        {
            _array_neurongroup_6_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_6_v,
                &_array_neurongroup_6_v[0],
                sizeof(_array_neurongroup_6_v[0])*_num__array_neurongroup_6_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_7_v; i++)
        {
            _array_neurongroup_7_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_7_v,
                &_array_neurongroup_7_v[0],
                sizeof(_array_neurongroup_7_v[0])*_num__array_neurongroup_7_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_8_v; i++)
        {
            _array_neurongroup_8_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_8_v,
                &_array_neurongroup_8_v[0],
                sizeof(_array_neurongroup_8_v[0])*_num__array_neurongroup_8_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__array_neurongroup_9_v; i++)
        {
            _array_neurongroup_9_v[i] = - 0.0706;
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_neurongroup_9_v,
                &_array_neurongroup_9_v[0],
                sizeof(_array_neurongroup_9_v[0])*_num__array_neurongroup_9_v,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__static_array__array_synapses_sources; i++)
        {
            _array_synapses_sources[i] = _static_array__array_synapses_sources[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_synapses_sources,
                &_array_synapses_sources[0],
                sizeof(_array_synapses_sources[0])*_num__array_synapses_sources,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__static_array__array_synapses_targets; i++)
        {
            _array_synapses_targets[i] = _static_array__array_synapses_targets[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_synapses_targets,
                &_array_synapses_targets[0],
                sizeof(_array_synapses_targets[0])*_num__array_synapses_targets,
                cudaMemcpyHostToDevice
            )
        );
        _run_synapses_synapses_create_array_codeobject();
        for(int i=0; i<_num__static_array__array_synapses_sources_1; i++)
        {
            _array_synapses_sources_1[i] = _static_array__array_synapses_sources_1[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_synapses_sources_1,
                &_array_synapses_sources_1[0],
                sizeof(_array_synapses_sources_1[0])*_num__array_synapses_sources_1,
                cudaMemcpyHostToDevice
            )
        );
        for(int i=0; i<_num__static_array__array_synapses_targets_1; i++)
        {
            _array_synapses_targets_1[i] = _static_array__array_synapses_targets_1[i];
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dev_array_synapses_targets_1,
                &_array_synapses_targets_1[0],
                sizeof(_array_synapses_targets_1[0])*_num__array_synapses_targets_1,
                cudaMemcpyHostToDevice
            )
        );
        _run_synapses_synapses_create_array_codeobject_1();
        _run_synapses_1_synapses_create_generator_codeobject();
        _run_synapses_1_group_variable_set_conditional_codeobject();
        _run_synapses_2_synapses_create_generator_codeobject();
        _run_synapses_2_group_variable_set_conditional_codeobject();
        _run_synapses_3_synapses_create_generator_codeobject();
        _run_synapses_3_group_variable_set_conditional_codeobject();
        _run_synapses_4_synapses_create_generator_codeobject();
        _run_synapses_4_group_variable_set_conditional_codeobject();
        _run_synapses_5_synapses_create_generator_codeobject();
        _run_synapses_5_group_variable_set_conditional_codeobject();
        _run_synapses_6_synapses_create_generator_codeobject();
        _run_synapses_6_group_variable_set_conditional_codeobject();
        _run_synapses_7_synapses_create_generator_codeobject();
        _run_synapses_7_group_variable_set_conditional_codeobject();
        _run_synapses_8_synapses_create_generator_codeobject();
        _run_synapses_8_group_variable_set_conditional_codeobject();
        _run_synapses_9_synapses_create_generator_codeobject();
        _run_synapses_9_group_variable_set_conditional_codeobject();
        _run_synapses_10_synapses_create_generator_codeobject();
        _run_synapses_10_group_variable_set_conditional_codeobject();
        _run_synapses_11_synapses_create_generator_codeobject();
        _run_synapses_11_group_variable_set_conditional_codeobject();
        _run_synapses_12_synapses_create_generator_codeobject();
        _run_synapses_12_group_variable_set_conditional_codeobject();
        _run_synapses_13_synapses_create_generator_codeobject();
        _run_synapses_13_group_variable_set_conditional_codeobject();
        _run_synapses_14_synapses_create_generator_codeobject();
        _run_synapses_14_group_variable_set_conditional_codeobject();
        _run_synapses_15_synapses_create_generator_codeobject();
        _run_synapses_15_group_variable_set_conditional_codeobject();
        _array_statemonitor__indices[0] = 0;
                            CUDA_SAFE_CALL(
                                cudaMemcpy(
                                    dev_array_statemonitor__indices + 0,
                                    &_array_statemonitor__indices[0],
                                    sizeof(_array_statemonitor__indices[0]),
                                    cudaMemcpyHostToDevice
                                )
                            );
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_synapses_10_pre_push_spikes();
        _before_run_synapses_11_pre_push_spikes();
        _before_run_synapses_12_pre_push_spikes();
        _before_run_synapses_13_pre_push_spikes();
        _before_run_synapses_14_pre_push_spikes();
        _before_run_synapses_15_pre_push_spikes();
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_3_pre_push_spikes();
        _before_run_synapses_4_pre_push_spikes();
        _before_run_synapses_5_pre_push_spikes();
        _before_run_synapses_6_pre_push_spikes();
        _before_run_synapses_7_pre_push_spikes();
        _before_run_synapses_8_pre_push_spikes();
        _before_run_synapses_9_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        dev_dynamic_array_synapses_delay.clear();
        dev_dynamic_array_synapses_delay.shrink_to_fit();
        dev_dynamic_array_synapses_1_delay.clear();
        dev_dynamic_array_synapses_1_delay.shrink_to_fit();
        dev_dynamic_array_synapses_2_delay.clear();
        dev_dynamic_array_synapses_2_delay.shrink_to_fit();
        dev_dynamic_array_synapses_3_delay.clear();
        dev_dynamic_array_synapses_3_delay.shrink_to_fit();
        dev_dynamic_array_synapses_4_delay.clear();
        dev_dynamic_array_synapses_4_delay.shrink_to_fit();
        dev_dynamic_array_synapses_5_delay.clear();
        dev_dynamic_array_synapses_5_delay.shrink_to_fit();
        dev_dynamic_array_synapses_6_delay.clear();
        dev_dynamic_array_synapses_6_delay.shrink_to_fit();
        dev_dynamic_array_synapses_7_delay.clear();
        dev_dynamic_array_synapses_7_delay.shrink_to_fit();
        dev_dynamic_array_synapses_8_delay.clear();
        dev_dynamic_array_synapses_8_delay.shrink_to_fit();
        dev_dynamic_array_synapses_9_delay.clear();
        dev_dynamic_array_synapses_9_delay.shrink_to_fit();
        dev_dynamic_array_synapses_10_delay.clear();
        dev_dynamic_array_synapses_10_delay.shrink_to_fit();
        dev_dynamic_array_synapses_11_delay.clear();
        dev_dynamic_array_synapses_11_delay.shrink_to_fit();
        dev_dynamic_array_synapses_12_delay.clear();
        dev_dynamic_array_synapses_12_delay.shrink_to_fit();
        dev_dynamic_array_synapses_13_delay.clear();
        dev_dynamic_array_synapses_13_delay.shrink_to_fit();
        dev_dynamic_array_synapses_14_delay.clear();
        dev_dynamic_array_synapses_14_delay.shrink_to_fit();
        dev_dynamic_array_synapses_15_delay.clear();
        dev_dynamic_array_synapses_15_delay.shrink_to_fit();
        network.clear();
        network.add(&defaultclock, _run_random_number_buffer);
        network.add(&defaultclock, _run_statemonitor_codeobject);
        network.add(&defaultclock, _run_synapses_10_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_11_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_12_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_13_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_14_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_15_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_1_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_2_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_3_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_4_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_5_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_6_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_7_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_8_summed_variable_Iinh_post_codeobject);
        network.add(&defaultclock, _run_synapses_9_summed_variable_Iexc_post_codeobject);
        network.add(&defaultclock, _run_synapses_summed_variable_Iexc2_post_codeobject);
        network.add(&defaultclock, _run_neurongroup_10_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_11_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_2_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_3_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_4_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_5_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_6_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_7_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_8_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_9_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_10_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_11_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_2_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_3_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_4_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_5_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_6_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_7_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_8_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_9_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_spikemonitor_codeobject);
        network.add(&defaultclock, _run_spikemonitor_1_codeobject);
        network.add(&defaultclock, _run_synapses_10_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_10_pre_codeobject);
        network.add(&defaultclock, _run_synapses_11_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_11_pre_codeobject);
        network.add(&defaultclock, _run_synapses_12_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_12_pre_codeobject);
        network.add(&defaultclock, _run_synapses_13_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_13_pre_codeobject);
        network.add(&defaultclock, _run_synapses_14_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_14_pre_codeobject);
        network.add(&defaultclock, _run_synapses_15_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_15_pre_codeobject);
        network.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_1_pre_codeobject);
        network.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_2_pre_codeobject);
        network.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_3_pre_codeobject);
        network.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_4_pre_codeobject);
        network.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_5_pre_codeobject);
        network.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_6_pre_codeobject);
        network.add(&defaultclock, _run_synapses_7_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_7_pre_codeobject);
        network.add(&defaultclock, _run_synapses_8_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_8_pre_codeobject);
        network.add(&defaultclock, _run_synapses_9_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_9_pre_codeobject);
        network.add(&defaultclock, _run_synapses_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_pre_codeobject);
        network.add(&defaultclock, _run_neurongroup_10_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_11_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_2_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_3_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_4_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_5_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_6_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_7_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_8_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_9_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject);
        set_from_command_line(args);
        network.run(4.0, NULL, 10.0);
        random_number_buffer.run_finished();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaProfilerStop());
        _after_run_neurongroup_10_spike_thresholder_codeobject();
        _after_run_neurongroup_11_spike_thresholder_codeobject();
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _after_run_neurongroup_2_spike_thresholder_codeobject();
        _after_run_neurongroup_3_spike_thresholder_codeobject();
        _after_run_neurongroup_4_spike_thresholder_codeobject();
        _after_run_neurongroup_5_spike_thresholder_codeobject();
        _after_run_neurongroup_6_spike_thresholder_codeobject();
        _after_run_neurongroup_7_spike_thresholder_codeobject();
        _after_run_neurongroup_8_spike_thresholder_codeobject();
        _after_run_neurongroup_9_spike_thresholder_codeobject();
        _after_run_neurongroup_spike_thresholder_codeobject();
        _debugmsg_spikemonitor_codeobject();
        
        _debugmsg_spikemonitor_1_codeobject();
        
        _debugmsg_synapses_10_pre_codeobject();
        
        _debugmsg_synapses_11_pre_codeobject();
        
        _debugmsg_synapses_12_pre_codeobject();
        
        _debugmsg_synapses_13_pre_codeobject();
        
        _debugmsg_synapses_14_pre_codeobject();
        
        _debugmsg_synapses_15_pre_codeobject();
        
        _debugmsg_synapses_1_pre_codeobject();
        
        _debugmsg_synapses_2_pre_codeobject();
        
        _debugmsg_synapses_3_pre_codeobject();
        
        _debugmsg_synapses_4_pre_codeobject();
        
        _debugmsg_synapses_5_pre_codeobject();
        
        _debugmsg_synapses_6_pre_codeobject();
        
        _debugmsg_synapses_7_pre_codeobject();
        
        _debugmsg_synapses_8_pre_codeobject();
        
        _debugmsg_synapses_9_pre_codeobject();
        
        _debugmsg_synapses_pre_codeobject();

    }

    //const double _run_time3 = (double)(std::clock() -_start_time3)/CLOCKS_PER_SEC;
    //printf("INFO: main_lines took %f seconds\n", _run_time3);

        

    brian_end();
        


    // Profiling
    //const double _run_time = (double)(std::clock() -_start_time)/CLOCKS_PER_SEC;
    //printf("INFO: main function took %f seconds\n", _run_time);

    return 0;
}