#include<stdlib.h>
#include "brianlib/cuda_utils.h"
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_10_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_10_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_10_stateupdater_codeobject.h"
#include "code_objects/neurongroup_11_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_11_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_11_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_2_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_2_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_2_stateupdater_codeobject.h"
#include "code_objects/neurongroup_3_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_3_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_3_stateupdater_codeobject.h"
#include "code_objects/neurongroup_4_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_4_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_4_stateupdater_codeobject.h"
#include "code_objects/neurongroup_5_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_5_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_5_stateupdater_codeobject.h"
#include "code_objects/neurongroup_6_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_6_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_6_stateupdater_codeobject.h"
#include "code_objects/neurongroup_7_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_7_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_7_stateupdater_codeobject.h"
#include "code_objects/neurongroup_8_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_8_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_8_stateupdater_codeobject.h"
#include "code_objects/neurongroup_9_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_9_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_9_stateupdater_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_10_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_10_pre_codeobject.h"
#include "code_objects/synapses_10_pre_push_spikes.h"
#include "code_objects/synapses_10_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_10_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_11_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_11_pre_codeobject.h"
#include "code_objects/synapses_11_pre_push_spikes.h"
#include "code_objects/synapses_11_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_11_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_12_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_12_pre_codeobject.h"
#include "code_objects/synapses_12_pre_push_spikes.h"
#include "code_objects/synapses_12_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_12_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_13_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_13_pre_codeobject.h"
#include "code_objects/synapses_13_pre_push_spikes.h"
#include "code_objects/synapses_13_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_13_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_14_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_14_pre_codeobject.h"
#include "code_objects/synapses_14_pre_push_spikes.h"
#include "code_objects/synapses_14_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_14_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_15_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_15_pre_codeobject.h"
#include "code_objects/synapses_15_pre_push_spikes.h"
#include "code_objects/synapses_15_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_15_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_2_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_3_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_3_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_4_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_4_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_5_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_7_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_7_pre_codeobject.h"
#include "code_objects/synapses_7_pre_push_spikes.h"
#include "code_objects/synapses_7_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_7_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_8_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_8_pre_codeobject.h"
#include "code_objects/synapses_8_pre_push_spikes.h"
#include "code_objects/synapses_8_summed_variable_Iinh_post_codeobject.h"
#include "code_objects/synapses_8_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_9_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_9_pre_codeobject.h"
#include "code_objects/synapses_9_pre_push_spikes.h"
#include "code_objects/synapses_9_summed_variable_Iexc_post_codeobject.h"
#include "code_objects/synapses_9_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_summed_variable_Iexc2_post_codeobject.h"
#include "code_objects/synapses_synapses_create_array_codeobject.h"
#include "code_objects/synapses_synapses_create_array_codeobject_1.h"


void brian_start()
{
    _init_arrays();
    _load_arrays();
    srand(time(NULL));

    // Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
}

void brian_end()
{
    _write_arrays();
    _dealloc_arrays();
}


