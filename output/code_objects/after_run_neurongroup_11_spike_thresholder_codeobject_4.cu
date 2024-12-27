#include "code_objects/neurongroup_11_spike_thresholder_codeobject_4.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/cuda_utils.h"
#include "brianlib/stdint_compat.h"
#include <cmath>
#include <stdint.h>
#include <ctime>
#include <stdio.h>


void _after_run_neurongroup_11_spike_thresholder_codeobject_4()
{
    using namespace brian;

///// HOST_CONSTANTS ///////////
const int64_t N = 120;
		const int _num_spikespace = 121;
		const double mV = 0.001;
		const int _numv = 120;

const int _N = N;
int num_threads, num_blocks;
int min_num_threads; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

CUDA_SAFE_CALL(
        cudaOccupancyMaxPotentialBlockSize(&min_num_threads, &num_threads,
            _reset_neurongroup_11_spike_thresholder_codeobject_4, 0, 0)  // last args: dynamicSMemSize, blockSizeLimit
        );

// Round up according to array size
num_blocks = (_N + num_threads - 1) / num_threads;

_reset_neurongroup_11_spike_thresholder_codeobject_4<<<num_blocks, num_threads>>>(
        dev_array_neurongroup_11__spikespace[current_idx_array_neurongroup_11__spikespace]
    );

CUDA_CHECK_ERROR("_reset_neurongroup_11_spike_thresholder_codeobject_4");
}
