

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/cuda_utils.h"
#include "network.h"
#include "rand.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <utility>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

size_t brian::used_device_memory = 0;
std::string brian::results_dir = "results/";  // can be overwritten by --results_dir command line arg

//////////////// clocks ///////////////////
Clock brian::defaultclock;

//////////////// networks /////////////////
Network brian::network;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void brian::set_variable_by_name(std::string name, std::string s_value) {
	size_t var_size;
	size_t data_size;
	std::for_each(s_value.begin(), s_value.end(), [](char& c) // modify in-place
    {
        c = std::tolower(static_cast<unsigned char>(c));
    });
    if (s_value == "true")
        s_value = "1";
    else if (s_value == "false")
        s_value = "0";
	// non-dynamic arrays
    if (name == "neurongroup_10.A") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_A,
                &brian::_array_neurongroup_10_A[0],
                sizeof(brian::_array_neurongroup_10_A[0])*brian::_num__array_neurongroup_10_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.batch_sum_X") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_batch_sum_X,
                &brian::_array_neurongroup_10_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_10_batch_sum_X[0])*brian::_num__array_neurongroup_10_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.Iexc") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_Iexc,
                &brian::_array_neurongroup_10_Iexc[0],
                sizeof(brian::_array_neurongroup_10_Iexc[0])*brian::_num__array_neurongroup_10_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.Iinh") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_Iinh,
                &brian::_array_neurongroup_10_Iinh[0],
                sizeof(brian::_array_neurongroup_10_Iinh[0])*brian::_num__array_neurongroup_10_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.running_sum_X") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_running_sum_X,
                &brian::_array_neurongroup_10_running_sum_X[0],
                sizeof(brian::_array_neurongroup_10_running_sum_X[0])*brian::_num__array_neurongroup_10_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.v") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_v,
                &brian::_array_neurongroup_10_v[0],
                sizeof(brian::_array_neurongroup_10_v[0])*brian::_num__array_neurongroup_10_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.X") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_X,
                &brian::_array_neurongroup_10_X[0],
                sizeof(brian::_array_neurongroup_10_X[0])*brian::_num__array_neurongroup_10_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_10.Y") {
        var_size = 16;
        data_size = 16*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_10_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_10_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_10_Y,
                &brian::_array_neurongroup_10_Y[0],
                sizeof(brian::_array_neurongroup_10_Y[0])*brian::_num__array_neurongroup_10_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.A") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_A,
                &brian::_array_neurongroup_11_A[0],
                sizeof(brian::_array_neurongroup_11_A[0])*brian::_num__array_neurongroup_11_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.Iexc") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_Iexc,
                &brian::_array_neurongroup_11_Iexc[0],
                sizeof(brian::_array_neurongroup_11_Iexc[0])*brian::_num__array_neurongroup_11_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.Iexc2") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_Iexc2, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_Iexc2, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_Iexc2,
                &brian::_array_neurongroup_11_Iexc2[0],
                sizeof(brian::_array_neurongroup_11_Iexc2[0])*brian::_num__array_neurongroup_11_Iexc2,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.indices") {
        var_size = 120;
        data_size = 120*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, brian::_array_neurongroup_11_indices, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_indices, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_indices,
                &brian::_array_neurongroup_11_indices[0],
                sizeof(brian::_array_neurongroup_11_indices[0])*brian::_num__array_neurongroup_11_indices,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.v") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_v,
                &brian::_array_neurongroup_11_v[0],
                sizeof(brian::_array_neurongroup_11_v[0])*brian::_num__array_neurongroup_11_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.X") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_X,
                &brian::_array_neurongroup_11_X[0],
                sizeof(brian::_array_neurongroup_11_X[0])*brian::_num__array_neurongroup_11_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_11.Y") {
        var_size = 120;
        data_size = 120*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_11_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_11_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_11_Y,
                &brian::_array_neurongroup_11_Y[0],
                sizeof(brian::_array_neurongroup_11_Y[0])*brian::_num__array_neurongroup_11_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_A,
                &brian::_array_neurongroup_1_A[0],
                sizeof(brian::_array_neurongroup_1_A[0])*brian::_num__array_neurongroup_1_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_batch_sum_X,
                &brian::_array_neurongroup_1_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_1_batch_sum_X[0])*brian::_num__array_neurongroup_1_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_Iexc,
                &brian::_array_neurongroup_1_Iexc[0],
                sizeof(brian::_array_neurongroup_1_Iexc[0])*brian::_num__array_neurongroup_1_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_Iinh,
                &brian::_array_neurongroup_1_Iinh[0],
                sizeof(brian::_array_neurongroup_1_Iinh[0])*brian::_num__array_neurongroup_1_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_running_sum_X,
                &brian::_array_neurongroup_1_running_sum_X[0],
                sizeof(brian::_array_neurongroup_1_running_sum_X[0])*brian::_num__array_neurongroup_1_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_v,
                &brian::_array_neurongroup_1_v[0],
                sizeof(brian::_array_neurongroup_1_v[0])*brian::_num__array_neurongroup_1_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_X,
                &brian::_array_neurongroup_1_X[0],
                sizeof(brian::_array_neurongroup_1_X[0])*brian::_num__array_neurongroup_1_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_1.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_1_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_1_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_1_Y,
                &brian::_array_neurongroup_1_Y[0],
                sizeof(brian::_array_neurongroup_1_Y[0])*brian::_num__array_neurongroup_1_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_A,
                &brian::_array_neurongroup_2_A[0],
                sizeof(brian::_array_neurongroup_2_A[0])*brian::_num__array_neurongroup_2_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_batch_sum_X,
                &brian::_array_neurongroup_2_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_2_batch_sum_X[0])*brian::_num__array_neurongroup_2_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_Iexc,
                &brian::_array_neurongroup_2_Iexc[0],
                sizeof(brian::_array_neurongroup_2_Iexc[0])*brian::_num__array_neurongroup_2_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_Iinh,
                &brian::_array_neurongroup_2_Iinh[0],
                sizeof(brian::_array_neurongroup_2_Iinh[0])*brian::_num__array_neurongroup_2_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_running_sum_X,
                &brian::_array_neurongroup_2_running_sum_X[0],
                sizeof(brian::_array_neurongroup_2_running_sum_X[0])*brian::_num__array_neurongroup_2_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_v,
                &brian::_array_neurongroup_2_v[0],
                sizeof(brian::_array_neurongroup_2_v[0])*brian::_num__array_neurongroup_2_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_X,
                &brian::_array_neurongroup_2_X[0],
                sizeof(brian::_array_neurongroup_2_X[0])*brian::_num__array_neurongroup_2_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_2.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_2_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_2_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_2_Y,
                &brian::_array_neurongroup_2_Y[0],
                sizeof(brian::_array_neurongroup_2_Y[0])*brian::_num__array_neurongroup_2_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_A,
                &brian::_array_neurongroup_3_A[0],
                sizeof(brian::_array_neurongroup_3_A[0])*brian::_num__array_neurongroup_3_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_batch_sum_X,
                &brian::_array_neurongroup_3_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_3_batch_sum_X[0])*brian::_num__array_neurongroup_3_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_Iexc,
                &brian::_array_neurongroup_3_Iexc[0],
                sizeof(brian::_array_neurongroup_3_Iexc[0])*brian::_num__array_neurongroup_3_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_Iinh,
                &brian::_array_neurongroup_3_Iinh[0],
                sizeof(brian::_array_neurongroup_3_Iinh[0])*brian::_num__array_neurongroup_3_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_running_sum_X,
                &brian::_array_neurongroup_3_running_sum_X[0],
                sizeof(brian::_array_neurongroup_3_running_sum_X[0])*brian::_num__array_neurongroup_3_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_v,
                &brian::_array_neurongroup_3_v[0],
                sizeof(brian::_array_neurongroup_3_v[0])*brian::_num__array_neurongroup_3_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_X,
                &brian::_array_neurongroup_3_X[0],
                sizeof(brian::_array_neurongroup_3_X[0])*brian::_num__array_neurongroup_3_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_3.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_3_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_3_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_3_Y,
                &brian::_array_neurongroup_3_Y[0],
                sizeof(brian::_array_neurongroup_3_Y[0])*brian::_num__array_neurongroup_3_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_A,
                &brian::_array_neurongroup_4_A[0],
                sizeof(brian::_array_neurongroup_4_A[0])*brian::_num__array_neurongroup_4_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_batch_sum_X,
                &brian::_array_neurongroup_4_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_4_batch_sum_X[0])*brian::_num__array_neurongroup_4_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_Iexc,
                &brian::_array_neurongroup_4_Iexc[0],
                sizeof(brian::_array_neurongroup_4_Iexc[0])*brian::_num__array_neurongroup_4_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_Iinh,
                &brian::_array_neurongroup_4_Iinh[0],
                sizeof(brian::_array_neurongroup_4_Iinh[0])*brian::_num__array_neurongroup_4_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_running_sum_X,
                &brian::_array_neurongroup_4_running_sum_X[0],
                sizeof(brian::_array_neurongroup_4_running_sum_X[0])*brian::_num__array_neurongroup_4_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_v,
                &brian::_array_neurongroup_4_v[0],
                sizeof(brian::_array_neurongroup_4_v[0])*brian::_num__array_neurongroup_4_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_X,
                &brian::_array_neurongroup_4_X[0],
                sizeof(brian::_array_neurongroup_4_X[0])*brian::_num__array_neurongroup_4_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_4.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_4_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_4_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_4_Y,
                &brian::_array_neurongroup_4_Y[0],
                sizeof(brian::_array_neurongroup_4_Y[0])*brian::_num__array_neurongroup_4_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_A,
                &brian::_array_neurongroup_5_A[0],
                sizeof(brian::_array_neurongroup_5_A[0])*brian::_num__array_neurongroup_5_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_batch_sum_X,
                &brian::_array_neurongroup_5_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_5_batch_sum_X[0])*brian::_num__array_neurongroup_5_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_Iexc,
                &brian::_array_neurongroup_5_Iexc[0],
                sizeof(brian::_array_neurongroup_5_Iexc[0])*brian::_num__array_neurongroup_5_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_Iinh,
                &brian::_array_neurongroup_5_Iinh[0],
                sizeof(brian::_array_neurongroup_5_Iinh[0])*brian::_num__array_neurongroup_5_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_running_sum_X,
                &brian::_array_neurongroup_5_running_sum_X[0],
                sizeof(brian::_array_neurongroup_5_running_sum_X[0])*brian::_num__array_neurongroup_5_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_v,
                &brian::_array_neurongroup_5_v[0],
                sizeof(brian::_array_neurongroup_5_v[0])*brian::_num__array_neurongroup_5_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_X,
                &brian::_array_neurongroup_5_X[0],
                sizeof(brian::_array_neurongroup_5_X[0])*brian::_num__array_neurongroup_5_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_5.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_5_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_5_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_5_Y,
                &brian::_array_neurongroup_5_Y[0],
                sizeof(brian::_array_neurongroup_5_Y[0])*brian::_num__array_neurongroup_5_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_A,
                &brian::_array_neurongroup_6_A[0],
                sizeof(brian::_array_neurongroup_6_A[0])*brian::_num__array_neurongroup_6_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_batch_sum_X,
                &brian::_array_neurongroup_6_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_6_batch_sum_X[0])*brian::_num__array_neurongroup_6_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_Iexc,
                &brian::_array_neurongroup_6_Iexc[0],
                sizeof(brian::_array_neurongroup_6_Iexc[0])*brian::_num__array_neurongroup_6_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_Iinh,
                &brian::_array_neurongroup_6_Iinh[0],
                sizeof(brian::_array_neurongroup_6_Iinh[0])*brian::_num__array_neurongroup_6_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_running_sum_X,
                &brian::_array_neurongroup_6_running_sum_X[0],
                sizeof(brian::_array_neurongroup_6_running_sum_X[0])*brian::_num__array_neurongroup_6_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_v,
                &brian::_array_neurongroup_6_v[0],
                sizeof(brian::_array_neurongroup_6_v[0])*brian::_num__array_neurongroup_6_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_X,
                &brian::_array_neurongroup_6_X[0],
                sizeof(brian::_array_neurongroup_6_X[0])*brian::_num__array_neurongroup_6_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_6.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_6_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_6_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_6_Y,
                &brian::_array_neurongroup_6_Y[0],
                sizeof(brian::_array_neurongroup_6_Y[0])*brian::_num__array_neurongroup_6_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_A,
                &brian::_array_neurongroup_7_A[0],
                sizeof(brian::_array_neurongroup_7_A[0])*brian::_num__array_neurongroup_7_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_batch_sum_X,
                &brian::_array_neurongroup_7_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_7_batch_sum_X[0])*brian::_num__array_neurongroup_7_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_Iexc,
                &brian::_array_neurongroup_7_Iexc[0],
                sizeof(brian::_array_neurongroup_7_Iexc[0])*brian::_num__array_neurongroup_7_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_Iinh,
                &brian::_array_neurongroup_7_Iinh[0],
                sizeof(brian::_array_neurongroup_7_Iinh[0])*brian::_num__array_neurongroup_7_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_running_sum_X,
                &brian::_array_neurongroup_7_running_sum_X[0],
                sizeof(brian::_array_neurongroup_7_running_sum_X[0])*brian::_num__array_neurongroup_7_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_v,
                &brian::_array_neurongroup_7_v[0],
                sizeof(brian::_array_neurongroup_7_v[0])*brian::_num__array_neurongroup_7_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_X,
                &brian::_array_neurongroup_7_X[0],
                sizeof(brian::_array_neurongroup_7_X[0])*brian::_num__array_neurongroup_7_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_7.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_7_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_7_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_7_Y,
                &brian::_array_neurongroup_7_Y[0],
                sizeof(brian::_array_neurongroup_7_Y[0])*brian::_num__array_neurongroup_7_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_A,
                &brian::_array_neurongroup_8_A[0],
                sizeof(brian::_array_neurongroup_8_A[0])*brian::_num__array_neurongroup_8_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_batch_sum_X,
                &brian::_array_neurongroup_8_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_8_batch_sum_X[0])*brian::_num__array_neurongroup_8_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_Iexc,
                &brian::_array_neurongroup_8_Iexc[0],
                sizeof(brian::_array_neurongroup_8_Iexc[0])*brian::_num__array_neurongroup_8_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_Iinh,
                &brian::_array_neurongroup_8_Iinh[0],
                sizeof(brian::_array_neurongroup_8_Iinh[0])*brian::_num__array_neurongroup_8_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_running_sum_X,
                &brian::_array_neurongroup_8_running_sum_X[0],
                sizeof(brian::_array_neurongroup_8_running_sum_X[0])*brian::_num__array_neurongroup_8_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_v,
                &brian::_array_neurongroup_8_v[0],
                sizeof(brian::_array_neurongroup_8_v[0])*brian::_num__array_neurongroup_8_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_X,
                &brian::_array_neurongroup_8_X[0],
                sizeof(brian::_array_neurongroup_8_X[0])*brian::_num__array_neurongroup_8_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_8.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_8_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_8_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_8_Y,
                &brian::_array_neurongroup_8_Y[0],
                sizeof(brian::_array_neurongroup_8_Y[0])*brian::_num__array_neurongroup_8_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_A,
                &brian::_array_neurongroup_9_A[0],
                sizeof(brian::_array_neurongroup_9_A[0])*brian::_num__array_neurongroup_9_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.batch_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_batch_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_batch_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_batch_sum_X,
                &brian::_array_neurongroup_9_batch_sum_X[0],
                sizeof(brian::_array_neurongroup_9_batch_sum_X[0])*brian::_num__array_neurongroup_9_batch_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.Iexc") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_Iexc, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_Iexc, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_Iexc,
                &brian::_array_neurongroup_9_Iexc[0],
                sizeof(brian::_array_neurongroup_9_Iexc[0])*brian::_num__array_neurongroup_9_Iexc,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.Iinh") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_Iinh, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_Iinh, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_Iinh,
                &brian::_array_neurongroup_9_Iinh[0],
                sizeof(brian::_array_neurongroup_9_Iinh[0])*brian::_num__array_neurongroup_9_Iinh,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.running_sum_X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_running_sum_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_running_sum_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_running_sum_X,
                &brian::_array_neurongroup_9_running_sum_X[0],
                sizeof(brian::_array_neurongroup_9_running_sum_X[0])*brian::_num__array_neurongroup_9_running_sum_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_v,
                &brian::_array_neurongroup_9_v[0],
                sizeof(brian::_array_neurongroup_9_v[0])*brian::_num__array_neurongroup_9_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_X,
                &brian::_array_neurongroup_9_X[0],
                sizeof(brian::_array_neurongroup_9_X[0])*brian::_num__array_neurongroup_9_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup_9.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_9_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_9_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_9_Y,
                &brian::_array_neurongroup_9_Y[0],
                sizeof(brian::_array_neurongroup_9_Y[0])*brian::_num__array_neurongroup_9_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup.A") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_A, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_A, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_A,
                &brian::_array_neurongroup_A[0],
                sizeof(brian::_array_neurongroup_A[0])*brian::_num__array_neurongroup_A,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup.indices") {
        var_size = 2;
        data_size = 2*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, brian::_array_neurongroup_indices, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_indices, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_indices,
                &brian::_array_neurongroup_indices[0],
                sizeof(brian::_array_neurongroup_indices[0])*brian::_num__array_neurongroup_indices,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup.v") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_v, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_v,
                &brian::_array_neurongroup_v[0],
                sizeof(brian::_array_neurongroup_v[0])*brian::_num__array_neurongroup_v,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup.X") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_X, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_X, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_X,
                &brian::_array_neurongroup_X[0],
                sizeof(brian::_array_neurongroup_X[0])*brian::_num__array_neurongroup_X,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "neurongroup.Y") {
        var_size = 2;
        data_size = 2*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_array_neurongroup_Y, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, brian::_array_neurongroup_Y, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_array_neurongroup_Y,
                &brian::_array_neurongroup_Y[0],
                sizeof(brian::_array_neurongroup_Y[0])*brian::_num__array_neurongroup_Y,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses_10.delay") {
        var_size = brian::_dynamic_array_synapses_10_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_10_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_10_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_10_delay[0]),
                &brian::_dynamic_array_synapses_10_delay[0],
                sizeof(brian::_dynamic_array_synapses_10_delay[0])*brian::_dynamic_array_synapses_10_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_10.w") {
        var_size = brian::_dynamic_array_synapses_10_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_10_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_10_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_10_w[0]),
                &brian::_dynamic_array_synapses_10_w[0],
                sizeof(brian::_dynamic_array_synapses_10_w[0])*brian::_dynamic_array_synapses_10_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_11.delay") {
        var_size = brian::_dynamic_array_synapses_11_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_11_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_11_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_11_delay[0]),
                &brian::_dynamic_array_synapses_11_delay[0],
                sizeof(brian::_dynamic_array_synapses_11_delay[0])*brian::_dynamic_array_synapses_11_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_11.w") {
        var_size = brian::_dynamic_array_synapses_11_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_11_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_11_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_11_w[0]),
                &brian::_dynamic_array_synapses_11_w[0],
                sizeof(brian::_dynamic_array_synapses_11_w[0])*brian::_dynamic_array_synapses_11_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_12.delay") {
        var_size = brian::_dynamic_array_synapses_12_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_12_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_12_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_12_delay[0]),
                &brian::_dynamic_array_synapses_12_delay[0],
                sizeof(brian::_dynamic_array_synapses_12_delay[0])*brian::_dynamic_array_synapses_12_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_12.w") {
        var_size = brian::_dynamic_array_synapses_12_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_12_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_12_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_12_w[0]),
                &brian::_dynamic_array_synapses_12_w[0],
                sizeof(brian::_dynamic_array_synapses_12_w[0])*brian::_dynamic_array_synapses_12_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_13.delay") {
        var_size = brian::_dynamic_array_synapses_13_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_13_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_13_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_13_delay[0]),
                &brian::_dynamic_array_synapses_13_delay[0],
                sizeof(brian::_dynamic_array_synapses_13_delay[0])*brian::_dynamic_array_synapses_13_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_13.w") {
        var_size = brian::_dynamic_array_synapses_13_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_13_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_13_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_13_w[0]),
                &brian::_dynamic_array_synapses_13_w[0],
                sizeof(brian::_dynamic_array_synapses_13_w[0])*brian::_dynamic_array_synapses_13_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_14.delay") {
        var_size = brian::_dynamic_array_synapses_14_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_14_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_14_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_14_delay[0]),
                &brian::_dynamic_array_synapses_14_delay[0],
                sizeof(brian::_dynamic_array_synapses_14_delay[0])*brian::_dynamic_array_synapses_14_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_14.w") {
        var_size = brian::_dynamic_array_synapses_14_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_14_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_14_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_14_w[0]),
                &brian::_dynamic_array_synapses_14_w[0],
                sizeof(brian::_dynamic_array_synapses_14_w[0])*brian::_dynamic_array_synapses_14_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_15.delay") {
        var_size = brian::_dynamic_array_synapses_15_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_15_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_15_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_15_delay[0]),
                &brian::_dynamic_array_synapses_15_delay[0],
                sizeof(brian::_dynamic_array_synapses_15_delay[0])*brian::_dynamic_array_synapses_15_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_15.w") {
        var_size = brian::_dynamic_array_synapses_15_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_15_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_15_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_15_w[0]),
                &brian::_dynamic_array_synapses_15_w[0],
                sizeof(brian::_dynamic_array_synapses_15_w[0])*brian::_dynamic_array_synapses_15_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_1.delay") {
        var_size = brian::_dynamic_array_synapses_1_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_1_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_1_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_1_delay[0]),
                &brian::_dynamic_array_synapses_1_delay[0],
                sizeof(brian::_dynamic_array_synapses_1_delay[0])*brian::_dynamic_array_synapses_1_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_1.w") {
        var_size = brian::_dynamic_array_synapses_1_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_1_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_1_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_1_w[0]),
                &brian::_dynamic_array_synapses_1_w[0],
                sizeof(brian::_dynamic_array_synapses_1_w[0])*brian::_dynamic_array_synapses_1_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_2.delay") {
        var_size = brian::_dynamic_array_synapses_2_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_2_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_2_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_2_delay[0]),
                &brian::_dynamic_array_synapses_2_delay[0],
                sizeof(brian::_dynamic_array_synapses_2_delay[0])*brian::_dynamic_array_synapses_2_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_2.w") {
        var_size = brian::_dynamic_array_synapses_2_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_2_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_2_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_2_w[0]),
                &brian::_dynamic_array_synapses_2_w[0],
                sizeof(brian::_dynamic_array_synapses_2_w[0])*brian::_dynamic_array_synapses_2_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_3.delay") {
        var_size = brian::_dynamic_array_synapses_3_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_3_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_3_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_3_delay[0]),
                &brian::_dynamic_array_synapses_3_delay[0],
                sizeof(brian::_dynamic_array_synapses_3_delay[0])*brian::_dynamic_array_synapses_3_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_3.w") {
        var_size = brian::_dynamic_array_synapses_3_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_3_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_3_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_3_w[0]),
                &brian::_dynamic_array_synapses_3_w[0],
                sizeof(brian::_dynamic_array_synapses_3_w[0])*brian::_dynamic_array_synapses_3_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_4.delay") {
        var_size = brian::_dynamic_array_synapses_4_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_4_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_4_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_4_delay[0]),
                &brian::_dynamic_array_synapses_4_delay[0],
                sizeof(brian::_dynamic_array_synapses_4_delay[0])*brian::_dynamic_array_synapses_4_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_4.w") {
        var_size = brian::_dynamic_array_synapses_4_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_4_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_4_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_4_w[0]),
                &brian::_dynamic_array_synapses_4_w[0],
                sizeof(brian::_dynamic_array_synapses_4_w[0])*brian::_dynamic_array_synapses_4_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_5.delay") {
        var_size = brian::_dynamic_array_synapses_5_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_5_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_5_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_5_delay[0]),
                &brian::_dynamic_array_synapses_5_delay[0],
                sizeof(brian::_dynamic_array_synapses_5_delay[0])*brian::_dynamic_array_synapses_5_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_5.w") {
        var_size = brian::_dynamic_array_synapses_5_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_5_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_5_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_5_w[0]),
                &brian::_dynamic_array_synapses_5_w[0],
                sizeof(brian::_dynamic_array_synapses_5_w[0])*brian::_dynamic_array_synapses_5_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_6.delay") {
        var_size = brian::_dynamic_array_synapses_6_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_6_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_6_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_6_delay[0]),
                &brian::_dynamic_array_synapses_6_delay[0],
                sizeof(brian::_dynamic_array_synapses_6_delay[0])*brian::_dynamic_array_synapses_6_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_6.w") {
        var_size = brian::_dynamic_array_synapses_6_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_6_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_6_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_6_w[0]),
                &brian::_dynamic_array_synapses_6_w[0],
                sizeof(brian::_dynamic_array_synapses_6_w[0])*brian::_dynamic_array_synapses_6_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_7.delay") {
        var_size = brian::_dynamic_array_synapses_7_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_7_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_7_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_7_delay[0]),
                &brian::_dynamic_array_synapses_7_delay[0],
                sizeof(brian::_dynamic_array_synapses_7_delay[0])*brian::_dynamic_array_synapses_7_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_7.w") {
        var_size = brian::_dynamic_array_synapses_7_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_7_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_7_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_7_w[0]),
                &brian::_dynamic_array_synapses_7_w[0],
                sizeof(brian::_dynamic_array_synapses_7_w[0])*brian::_dynamic_array_synapses_7_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_8.delay") {
        var_size = brian::_dynamic_array_synapses_8_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_8_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_8_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_8_delay[0]),
                &brian::_dynamic_array_synapses_8_delay[0],
                sizeof(brian::_dynamic_array_synapses_8_delay[0])*brian::_dynamic_array_synapses_8_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_8.w") {
        var_size = brian::_dynamic_array_synapses_8_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_8_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_8_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_8_w[0]),
                &brian::_dynamic_array_synapses_8_w[0],
                sizeof(brian::_dynamic_array_synapses_8_w[0])*brian::_dynamic_array_synapses_8_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_9.delay") {
        var_size = brian::_dynamic_array_synapses_9_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_9_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_9_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_9_delay[0]),
                &brian::_dynamic_array_synapses_9_delay[0],
                sizeof(brian::_dynamic_array_synapses_9_delay[0])*brian::_dynamic_array_synapses_9_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses_9.w") {
        var_size = brian::_dynamic_array_synapses_9_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_9_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_9_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_9_w[0]),
                &brian::_dynamic_array_synapses_9_w[0],
                sizeof(brian::_dynamic_array_synapses_9_w[0])*brian::_dynamic_array_synapses_9_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses.delay") {
        var_size = brian::_dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_delay[0]),
                &brian::_dynamic_array_synapses_delay[0],
                sizeof(brian::_dynamic_array_synapses_delay[0])*brian::_dynamic_array_synapses_delay.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "synapses.w") {
        var_size = brian::_dynamic_array_synapses_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &brian::_dynamic_array_synapses_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &brian::_dynamic_array_synapses_w[0], data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                thrust::raw_pointer_cast(&brian::dev_dynamic_array_synapses_w[0]),
                &brian::_dynamic_array_synapses_w[0],
                sizeof(brian::_dynamic_array_synapses_w[0])*brian::_dynamic_array_synapses_w.size(),
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "_timedarray.values") {
        var_size = 9000;
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_timedarray_values, var_size, (double)atof(s_value.c_str()));


        } else {
            // set from file
            set_variable_from_file(name, brian::_timedarray_values, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_timedarray_values,
                &brian::_timedarray_values[0],
                data_size,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    if (name == "_timedarray_2.values") {
        var_size = 4800;
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, brian::_timedarray_2_values, var_size, (double)atof(s_value.c_str()));


        } else {
            // set from file
            set_variable_from_file(name, brian::_timedarray_2_values, data_size, s_value);
        }
        // copy to device
        CUDA_SAFE_CALL(
            cudaMemcpy(
                brian::dev_timedarray_2_values,
                &brian::_timedarray_2_values[0],
                data_size,
                cudaMemcpyHostToDevice
            )
        );
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * brian::_array_defaultclock_dt;
double * brian::dev_array_defaultclock_dt;
__device__ double * brian::d_array_defaultclock_dt;
const int brian::_num__array_defaultclock_dt = 1;

double * brian::_array_defaultclock_t;
double * brian::dev_array_defaultclock_t;
__device__ double * brian::d_array_defaultclock_t;
const int brian::_num__array_defaultclock_t = 1;

int64_t * brian::_array_defaultclock_timestep;
int64_t * brian::dev_array_defaultclock_timestep;
__device__ int64_t * brian::d_array_defaultclock_timestep;
const int brian::_num__array_defaultclock_timestep = 1;

double * brian::_array_neurongroup_10_A;
double * brian::dev_array_neurongroup_10_A;
__device__ double * brian::d_array_neurongroup_10_A;
const int brian::_num__array_neurongroup_10_A = 16;

double * brian::_array_neurongroup_10_batch_sum_X;
double * brian::dev_array_neurongroup_10_batch_sum_X;
__device__ double * brian::d_array_neurongroup_10_batch_sum_X;
const int brian::_num__array_neurongroup_10_batch_sum_X = 16;

int32_t * brian::_array_neurongroup_10_i;
int32_t * brian::dev_array_neurongroup_10_i;
__device__ int32_t * brian::d_array_neurongroup_10_i;
const int brian::_num__array_neurongroup_10_i = 16;

double * brian::_array_neurongroup_10_Iexc;
double * brian::dev_array_neurongroup_10_Iexc;
__device__ double * brian::d_array_neurongroup_10_Iexc;
const int brian::_num__array_neurongroup_10_Iexc = 16;

double * brian::_array_neurongroup_10_Iinh;
double * brian::dev_array_neurongroup_10_Iinh;
__device__ double * brian::d_array_neurongroup_10_Iinh;
const int brian::_num__array_neurongroup_10_Iinh = 16;

double * brian::_array_neurongroup_10_running_sum_X;
double * brian::dev_array_neurongroup_10_running_sum_X;
__device__ double * brian::d_array_neurongroup_10_running_sum_X;
const int brian::_num__array_neurongroup_10_running_sum_X = 16;

double * brian::_array_neurongroup_10_v;
double * brian::dev_array_neurongroup_10_v;
__device__ double * brian::d_array_neurongroup_10_v;
const int brian::_num__array_neurongroup_10_v = 16;

double * brian::_array_neurongroup_10_X;
double * brian::dev_array_neurongroup_10_X;
__device__ double * brian::d_array_neurongroup_10_X;
const int brian::_num__array_neurongroup_10_X = 16;

double * brian::_array_neurongroup_10_Y;
double * brian::dev_array_neurongroup_10_Y;
__device__ double * brian::d_array_neurongroup_10_Y;
const int brian::_num__array_neurongroup_10_Y = 16;

double * brian::_array_neurongroup_11_A;
double * brian::dev_array_neurongroup_11_A;
__device__ double * brian::d_array_neurongroup_11_A;
const int brian::_num__array_neurongroup_11_A = 120;

int32_t * brian::_array_neurongroup_11_i;
int32_t * brian::dev_array_neurongroup_11_i;
__device__ int32_t * brian::d_array_neurongroup_11_i;
const int brian::_num__array_neurongroup_11_i = 120;

double * brian::_array_neurongroup_11_Iexc;
double * brian::dev_array_neurongroup_11_Iexc;
__device__ double * brian::d_array_neurongroup_11_Iexc;
const int brian::_num__array_neurongroup_11_Iexc = 120;

double * brian::_array_neurongroup_11_Iexc2;
double * brian::dev_array_neurongroup_11_Iexc2;
__device__ double * brian::d_array_neurongroup_11_Iexc2;
const int brian::_num__array_neurongroup_11_Iexc2 = 120;

int32_t * brian::_array_neurongroup_11_indices;
int32_t * brian::dev_array_neurongroup_11_indices;
__device__ int32_t * brian::d_array_neurongroup_11_indices;
const int brian::_num__array_neurongroup_11_indices = 120;

double * brian::_array_neurongroup_11_v;
double * brian::dev_array_neurongroup_11_v;
__device__ double * brian::d_array_neurongroup_11_v;
const int brian::_num__array_neurongroup_11_v = 120;

double * brian::_array_neurongroup_11_X;
double * brian::dev_array_neurongroup_11_X;
__device__ double * brian::d_array_neurongroup_11_X;
const int brian::_num__array_neurongroup_11_X = 120;

double * brian::_array_neurongroup_11_Y;
double * brian::dev_array_neurongroup_11_Y;
__device__ double * brian::d_array_neurongroup_11_Y;
const int brian::_num__array_neurongroup_11_Y = 120;

double * brian::_array_neurongroup_1_A;
double * brian::dev_array_neurongroup_1_A;
__device__ double * brian::d_array_neurongroup_1_A;
const int brian::_num__array_neurongroup_1_A = 2;

double * brian::_array_neurongroup_1_batch_sum_X;
double * brian::dev_array_neurongroup_1_batch_sum_X;
__device__ double * brian::d_array_neurongroup_1_batch_sum_X;
const int brian::_num__array_neurongroup_1_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_1_i;
int32_t * brian::dev_array_neurongroup_1_i;
__device__ int32_t * brian::d_array_neurongroup_1_i;
const int brian::_num__array_neurongroup_1_i = 2;

double * brian::_array_neurongroup_1_Iexc;
double * brian::dev_array_neurongroup_1_Iexc;
__device__ double * brian::d_array_neurongroup_1_Iexc;
const int brian::_num__array_neurongroup_1_Iexc = 2;

double * brian::_array_neurongroup_1_Iinh;
double * brian::dev_array_neurongroup_1_Iinh;
__device__ double * brian::d_array_neurongroup_1_Iinh;
const int brian::_num__array_neurongroup_1_Iinh = 2;

double * brian::_array_neurongroup_1_running_sum_X;
double * brian::dev_array_neurongroup_1_running_sum_X;
__device__ double * brian::d_array_neurongroup_1_running_sum_X;
const int brian::_num__array_neurongroup_1_running_sum_X = 2;

double * brian::_array_neurongroup_1_v;
double * brian::dev_array_neurongroup_1_v;
__device__ double * brian::d_array_neurongroup_1_v;
const int brian::_num__array_neurongroup_1_v = 2;

double * brian::_array_neurongroup_1_X;
double * brian::dev_array_neurongroup_1_X;
__device__ double * brian::d_array_neurongroup_1_X;
const int brian::_num__array_neurongroup_1_X = 2;

double * brian::_array_neurongroup_1_Y;
double * brian::dev_array_neurongroup_1_Y;
__device__ double * brian::d_array_neurongroup_1_Y;
const int brian::_num__array_neurongroup_1_Y = 2;

double * brian::_array_neurongroup_2_A;
double * brian::dev_array_neurongroup_2_A;
__device__ double * brian::d_array_neurongroup_2_A;
const int brian::_num__array_neurongroup_2_A = 2;

double * brian::_array_neurongroup_2_batch_sum_X;
double * brian::dev_array_neurongroup_2_batch_sum_X;
__device__ double * brian::d_array_neurongroup_2_batch_sum_X;
const int brian::_num__array_neurongroup_2_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_2_i;
int32_t * brian::dev_array_neurongroup_2_i;
__device__ int32_t * brian::d_array_neurongroup_2_i;
const int brian::_num__array_neurongroup_2_i = 2;

double * brian::_array_neurongroup_2_Iexc;
double * brian::dev_array_neurongroup_2_Iexc;
__device__ double * brian::d_array_neurongroup_2_Iexc;
const int brian::_num__array_neurongroup_2_Iexc = 2;

double * brian::_array_neurongroup_2_Iinh;
double * brian::dev_array_neurongroup_2_Iinh;
__device__ double * brian::d_array_neurongroup_2_Iinh;
const int brian::_num__array_neurongroup_2_Iinh = 2;

double * brian::_array_neurongroup_2_running_sum_X;
double * brian::dev_array_neurongroup_2_running_sum_X;
__device__ double * brian::d_array_neurongroup_2_running_sum_X;
const int brian::_num__array_neurongroup_2_running_sum_X = 2;

double * brian::_array_neurongroup_2_v;
double * brian::dev_array_neurongroup_2_v;
__device__ double * brian::d_array_neurongroup_2_v;
const int brian::_num__array_neurongroup_2_v = 2;

double * brian::_array_neurongroup_2_X;
double * brian::dev_array_neurongroup_2_X;
__device__ double * brian::d_array_neurongroup_2_X;
const int brian::_num__array_neurongroup_2_X = 2;

double * brian::_array_neurongroup_2_Y;
double * brian::dev_array_neurongroup_2_Y;
__device__ double * brian::d_array_neurongroup_2_Y;
const int brian::_num__array_neurongroup_2_Y = 2;

double * brian::_array_neurongroup_3_A;
double * brian::dev_array_neurongroup_3_A;
__device__ double * brian::d_array_neurongroup_3_A;
const int brian::_num__array_neurongroup_3_A = 2;

double * brian::_array_neurongroup_3_batch_sum_X;
double * brian::dev_array_neurongroup_3_batch_sum_X;
__device__ double * brian::d_array_neurongroup_3_batch_sum_X;
const int brian::_num__array_neurongroup_3_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_3_i;
int32_t * brian::dev_array_neurongroup_3_i;
__device__ int32_t * brian::d_array_neurongroup_3_i;
const int brian::_num__array_neurongroup_3_i = 2;

double * brian::_array_neurongroup_3_Iexc;
double * brian::dev_array_neurongroup_3_Iexc;
__device__ double * brian::d_array_neurongroup_3_Iexc;
const int brian::_num__array_neurongroup_3_Iexc = 2;

double * brian::_array_neurongroup_3_Iinh;
double * brian::dev_array_neurongroup_3_Iinh;
__device__ double * brian::d_array_neurongroup_3_Iinh;
const int brian::_num__array_neurongroup_3_Iinh = 2;

double * brian::_array_neurongroup_3_running_sum_X;
double * brian::dev_array_neurongroup_3_running_sum_X;
__device__ double * brian::d_array_neurongroup_3_running_sum_X;
const int brian::_num__array_neurongroup_3_running_sum_X = 2;

double * brian::_array_neurongroup_3_v;
double * brian::dev_array_neurongroup_3_v;
__device__ double * brian::d_array_neurongroup_3_v;
const int brian::_num__array_neurongroup_3_v = 2;

double * brian::_array_neurongroup_3_X;
double * brian::dev_array_neurongroup_3_X;
__device__ double * brian::d_array_neurongroup_3_X;
const int brian::_num__array_neurongroup_3_X = 2;

double * brian::_array_neurongroup_3_Y;
double * brian::dev_array_neurongroup_3_Y;
__device__ double * brian::d_array_neurongroup_3_Y;
const int brian::_num__array_neurongroup_3_Y = 2;

double * brian::_array_neurongroup_4_A;
double * brian::dev_array_neurongroup_4_A;
__device__ double * brian::d_array_neurongroup_4_A;
const int brian::_num__array_neurongroup_4_A = 2;

double * brian::_array_neurongroup_4_batch_sum_X;
double * brian::dev_array_neurongroup_4_batch_sum_X;
__device__ double * brian::d_array_neurongroup_4_batch_sum_X;
const int brian::_num__array_neurongroup_4_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_4_i;
int32_t * brian::dev_array_neurongroup_4_i;
__device__ int32_t * brian::d_array_neurongroup_4_i;
const int brian::_num__array_neurongroup_4_i = 2;

double * brian::_array_neurongroup_4_Iexc;
double * brian::dev_array_neurongroup_4_Iexc;
__device__ double * brian::d_array_neurongroup_4_Iexc;
const int brian::_num__array_neurongroup_4_Iexc = 2;

double * brian::_array_neurongroup_4_Iinh;
double * brian::dev_array_neurongroup_4_Iinh;
__device__ double * brian::d_array_neurongroup_4_Iinh;
const int brian::_num__array_neurongroup_4_Iinh = 2;

double * brian::_array_neurongroup_4_running_sum_X;
double * brian::dev_array_neurongroup_4_running_sum_X;
__device__ double * brian::d_array_neurongroup_4_running_sum_X;
const int brian::_num__array_neurongroup_4_running_sum_X = 2;

double * brian::_array_neurongroup_4_v;
double * brian::dev_array_neurongroup_4_v;
__device__ double * brian::d_array_neurongroup_4_v;
const int brian::_num__array_neurongroup_4_v = 2;

double * brian::_array_neurongroup_4_X;
double * brian::dev_array_neurongroup_4_X;
__device__ double * brian::d_array_neurongroup_4_X;
const int brian::_num__array_neurongroup_4_X = 2;

double * brian::_array_neurongroup_4_Y;
double * brian::dev_array_neurongroup_4_Y;
__device__ double * brian::d_array_neurongroup_4_Y;
const int brian::_num__array_neurongroup_4_Y = 2;

double * brian::_array_neurongroup_5_A;
double * brian::dev_array_neurongroup_5_A;
__device__ double * brian::d_array_neurongroup_5_A;
const int brian::_num__array_neurongroup_5_A = 2;

double * brian::_array_neurongroup_5_batch_sum_X;
double * brian::dev_array_neurongroup_5_batch_sum_X;
__device__ double * brian::d_array_neurongroup_5_batch_sum_X;
const int brian::_num__array_neurongroup_5_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_5_i;
int32_t * brian::dev_array_neurongroup_5_i;
__device__ int32_t * brian::d_array_neurongroup_5_i;
const int brian::_num__array_neurongroup_5_i = 2;

double * brian::_array_neurongroup_5_Iexc;
double * brian::dev_array_neurongroup_5_Iexc;
__device__ double * brian::d_array_neurongroup_5_Iexc;
const int brian::_num__array_neurongroup_5_Iexc = 2;

double * brian::_array_neurongroup_5_Iinh;
double * brian::dev_array_neurongroup_5_Iinh;
__device__ double * brian::d_array_neurongroup_5_Iinh;
const int brian::_num__array_neurongroup_5_Iinh = 2;

double * brian::_array_neurongroup_5_running_sum_X;
double * brian::dev_array_neurongroup_5_running_sum_X;
__device__ double * brian::d_array_neurongroup_5_running_sum_X;
const int brian::_num__array_neurongroup_5_running_sum_X = 2;

double * brian::_array_neurongroup_5_v;
double * brian::dev_array_neurongroup_5_v;
__device__ double * brian::d_array_neurongroup_5_v;
const int brian::_num__array_neurongroup_5_v = 2;

double * brian::_array_neurongroup_5_X;
double * brian::dev_array_neurongroup_5_X;
__device__ double * brian::d_array_neurongroup_5_X;
const int brian::_num__array_neurongroup_5_X = 2;

double * brian::_array_neurongroup_5_Y;
double * brian::dev_array_neurongroup_5_Y;
__device__ double * brian::d_array_neurongroup_5_Y;
const int brian::_num__array_neurongroup_5_Y = 2;

double * brian::_array_neurongroup_6_A;
double * brian::dev_array_neurongroup_6_A;
__device__ double * brian::d_array_neurongroup_6_A;
const int brian::_num__array_neurongroup_6_A = 2;

double * brian::_array_neurongroup_6_batch_sum_X;
double * brian::dev_array_neurongroup_6_batch_sum_X;
__device__ double * brian::d_array_neurongroup_6_batch_sum_X;
const int brian::_num__array_neurongroup_6_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_6_i;
int32_t * brian::dev_array_neurongroup_6_i;
__device__ int32_t * brian::d_array_neurongroup_6_i;
const int brian::_num__array_neurongroup_6_i = 2;

double * brian::_array_neurongroup_6_Iexc;
double * brian::dev_array_neurongroup_6_Iexc;
__device__ double * brian::d_array_neurongroup_6_Iexc;
const int brian::_num__array_neurongroup_6_Iexc = 2;

double * brian::_array_neurongroup_6_Iinh;
double * brian::dev_array_neurongroup_6_Iinh;
__device__ double * brian::d_array_neurongroup_6_Iinh;
const int brian::_num__array_neurongroup_6_Iinh = 2;

double * brian::_array_neurongroup_6_running_sum_X;
double * brian::dev_array_neurongroup_6_running_sum_X;
__device__ double * brian::d_array_neurongroup_6_running_sum_X;
const int brian::_num__array_neurongroup_6_running_sum_X = 2;

double * brian::_array_neurongroup_6_v;
double * brian::dev_array_neurongroup_6_v;
__device__ double * brian::d_array_neurongroup_6_v;
const int brian::_num__array_neurongroup_6_v = 2;

double * brian::_array_neurongroup_6_X;
double * brian::dev_array_neurongroup_6_X;
__device__ double * brian::d_array_neurongroup_6_X;
const int brian::_num__array_neurongroup_6_X = 2;

double * brian::_array_neurongroup_6_Y;
double * brian::dev_array_neurongroup_6_Y;
__device__ double * brian::d_array_neurongroup_6_Y;
const int brian::_num__array_neurongroup_6_Y = 2;

double * brian::_array_neurongroup_7_A;
double * brian::dev_array_neurongroup_7_A;
__device__ double * brian::d_array_neurongroup_7_A;
const int brian::_num__array_neurongroup_7_A = 2;

double * brian::_array_neurongroup_7_batch_sum_X;
double * brian::dev_array_neurongroup_7_batch_sum_X;
__device__ double * brian::d_array_neurongroup_7_batch_sum_X;
const int brian::_num__array_neurongroup_7_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_7_i;
int32_t * brian::dev_array_neurongroup_7_i;
__device__ int32_t * brian::d_array_neurongroup_7_i;
const int brian::_num__array_neurongroup_7_i = 2;

double * brian::_array_neurongroup_7_Iexc;
double * brian::dev_array_neurongroup_7_Iexc;
__device__ double * brian::d_array_neurongroup_7_Iexc;
const int brian::_num__array_neurongroup_7_Iexc = 2;

double * brian::_array_neurongroup_7_Iinh;
double * brian::dev_array_neurongroup_7_Iinh;
__device__ double * brian::d_array_neurongroup_7_Iinh;
const int brian::_num__array_neurongroup_7_Iinh = 2;

double * brian::_array_neurongroup_7_running_sum_X;
double * brian::dev_array_neurongroup_7_running_sum_X;
__device__ double * brian::d_array_neurongroup_7_running_sum_X;
const int brian::_num__array_neurongroup_7_running_sum_X = 2;

double * brian::_array_neurongroup_7_v;
double * brian::dev_array_neurongroup_7_v;
__device__ double * brian::d_array_neurongroup_7_v;
const int brian::_num__array_neurongroup_7_v = 2;

double * brian::_array_neurongroup_7_X;
double * brian::dev_array_neurongroup_7_X;
__device__ double * brian::d_array_neurongroup_7_X;
const int brian::_num__array_neurongroup_7_X = 2;

double * brian::_array_neurongroup_7_Y;
double * brian::dev_array_neurongroup_7_Y;
__device__ double * brian::d_array_neurongroup_7_Y;
const int brian::_num__array_neurongroup_7_Y = 2;

double * brian::_array_neurongroup_8_A;
double * brian::dev_array_neurongroup_8_A;
__device__ double * brian::d_array_neurongroup_8_A;
const int brian::_num__array_neurongroup_8_A = 2;

double * brian::_array_neurongroup_8_batch_sum_X;
double * brian::dev_array_neurongroup_8_batch_sum_X;
__device__ double * brian::d_array_neurongroup_8_batch_sum_X;
const int brian::_num__array_neurongroup_8_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_8_i;
int32_t * brian::dev_array_neurongroup_8_i;
__device__ int32_t * brian::d_array_neurongroup_8_i;
const int brian::_num__array_neurongroup_8_i = 2;

double * brian::_array_neurongroup_8_Iexc;
double * brian::dev_array_neurongroup_8_Iexc;
__device__ double * brian::d_array_neurongroup_8_Iexc;
const int brian::_num__array_neurongroup_8_Iexc = 2;

double * brian::_array_neurongroup_8_Iinh;
double * brian::dev_array_neurongroup_8_Iinh;
__device__ double * brian::d_array_neurongroup_8_Iinh;
const int brian::_num__array_neurongroup_8_Iinh = 2;

double * brian::_array_neurongroup_8_running_sum_X;
double * brian::dev_array_neurongroup_8_running_sum_X;
__device__ double * brian::d_array_neurongroup_8_running_sum_X;
const int brian::_num__array_neurongroup_8_running_sum_X = 2;

double * brian::_array_neurongroup_8_v;
double * brian::dev_array_neurongroup_8_v;
__device__ double * brian::d_array_neurongroup_8_v;
const int brian::_num__array_neurongroup_8_v = 2;

double * brian::_array_neurongroup_8_X;
double * brian::dev_array_neurongroup_8_X;
__device__ double * brian::d_array_neurongroup_8_X;
const int brian::_num__array_neurongroup_8_X = 2;

double * brian::_array_neurongroup_8_Y;
double * brian::dev_array_neurongroup_8_Y;
__device__ double * brian::d_array_neurongroup_8_Y;
const int brian::_num__array_neurongroup_8_Y = 2;

double * brian::_array_neurongroup_9_A;
double * brian::dev_array_neurongroup_9_A;
__device__ double * brian::d_array_neurongroup_9_A;
const int brian::_num__array_neurongroup_9_A = 2;

double * brian::_array_neurongroup_9_batch_sum_X;
double * brian::dev_array_neurongroup_9_batch_sum_X;
__device__ double * brian::d_array_neurongroup_9_batch_sum_X;
const int brian::_num__array_neurongroup_9_batch_sum_X = 2;

int32_t * brian::_array_neurongroup_9_i;
int32_t * brian::dev_array_neurongroup_9_i;
__device__ int32_t * brian::d_array_neurongroup_9_i;
const int brian::_num__array_neurongroup_9_i = 2;

double * brian::_array_neurongroup_9_Iexc;
double * brian::dev_array_neurongroup_9_Iexc;
__device__ double * brian::d_array_neurongroup_9_Iexc;
const int brian::_num__array_neurongroup_9_Iexc = 2;

double * brian::_array_neurongroup_9_Iinh;
double * brian::dev_array_neurongroup_9_Iinh;
__device__ double * brian::d_array_neurongroup_9_Iinh;
const int brian::_num__array_neurongroup_9_Iinh = 2;

double * brian::_array_neurongroup_9_running_sum_X;
double * brian::dev_array_neurongroup_9_running_sum_X;
__device__ double * brian::d_array_neurongroup_9_running_sum_X;
const int brian::_num__array_neurongroup_9_running_sum_X = 2;

double * brian::_array_neurongroup_9_v;
double * brian::dev_array_neurongroup_9_v;
__device__ double * brian::d_array_neurongroup_9_v;
const int brian::_num__array_neurongroup_9_v = 2;

double * brian::_array_neurongroup_9_X;
double * brian::dev_array_neurongroup_9_X;
__device__ double * brian::d_array_neurongroup_9_X;
const int brian::_num__array_neurongroup_9_X = 2;

double * brian::_array_neurongroup_9_Y;
double * brian::dev_array_neurongroup_9_Y;
__device__ double * brian::d_array_neurongroup_9_Y;
const int brian::_num__array_neurongroup_9_Y = 2;

double * brian::_array_neurongroup_A;
double * brian::dev_array_neurongroup_A;
__device__ double * brian::d_array_neurongroup_A;
const int brian::_num__array_neurongroup_A = 2;

int32_t * brian::_array_neurongroup_i;
int32_t * brian::dev_array_neurongroup_i;
__device__ int32_t * brian::d_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 2;

int32_t * brian::_array_neurongroup_indices;
int32_t * brian::dev_array_neurongroup_indices;
__device__ int32_t * brian::d_array_neurongroup_indices;
const int brian::_num__array_neurongroup_indices = 2;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
__device__ double * brian::d_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 2;

double * brian::_array_neurongroup_X;
double * brian::dev_array_neurongroup_X;
__device__ double * brian::d_array_neurongroup_X;
const int brian::_num__array_neurongroup_X = 2;

double * brian::_array_neurongroup_Y;
double * brian::dev_array_neurongroup_Y;
__device__ double * brian::d_array_neurongroup_Y;
const int brian::_num__array_neurongroup_Y = 2;

int32_t * brian::_array_spikemonitor__source_idx;
int32_t * brian::dev_array_spikemonitor__source_idx;
__device__ int32_t * brian::d_array_spikemonitor__source_idx;
const int brian::_num__array_spikemonitor__source_idx = 2;

int32_t * brian::_array_spikemonitor_count;
int32_t * brian::dev_array_spikemonitor_count;
__device__ int32_t * brian::d_array_spikemonitor_count;
const int brian::_num__array_spikemonitor_count = 2;

int32_t * brian::_array_spikemonitor_N;
int32_t * brian::dev_array_spikemonitor_N;
__device__ int32_t * brian::d_array_spikemonitor_N;
const int brian::_num__array_spikemonitor_N = 1;

int32_t * brian::_array_statemonitor__indices;
int32_t * brian::dev_array_statemonitor__indices;
__device__ int32_t * brian::d_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 1;

double * brian::_array_statemonitor_I;
double * brian::dev_array_statemonitor_I;
__device__ double * brian::d_array_statemonitor_I;
const int brian::_num__array_statemonitor_I = (0, 1);

int32_t * brian::_array_statemonitor_N;
int32_t * brian::dev_array_statemonitor_N;
__device__ int32_t * brian::d_array_statemonitor_N;
const int brian::_num__array_statemonitor_N = 1;

double * brian::_array_statemonitor_v;
double * brian::dev_array_statemonitor_v;
__device__ double * brian::d_array_statemonitor_v;
const int brian::_num__array_statemonitor_v = (0, 1);

int32_t * brian::_array_synapses_10_N;
int32_t * brian::dev_array_synapses_10_N;
__device__ int32_t * brian::d_array_synapses_10_N;
const int brian::_num__array_synapses_10_N = 1;

int32_t * brian::_array_synapses_11_N;
int32_t * brian::dev_array_synapses_11_N;
__device__ int32_t * brian::d_array_synapses_11_N;
const int brian::_num__array_synapses_11_N = 1;

int32_t * brian::_array_synapses_12_N;
int32_t * brian::dev_array_synapses_12_N;
__device__ int32_t * brian::d_array_synapses_12_N;
const int brian::_num__array_synapses_12_N = 1;

int32_t * brian::_array_synapses_13_N;
int32_t * brian::dev_array_synapses_13_N;
__device__ int32_t * brian::d_array_synapses_13_N;
const int brian::_num__array_synapses_13_N = 1;

int32_t * brian::_array_synapses_14_N;
int32_t * brian::dev_array_synapses_14_N;
__device__ int32_t * brian::d_array_synapses_14_N;
const int brian::_num__array_synapses_14_N = 1;

int32_t * brian::_array_synapses_15_N;
int32_t * brian::dev_array_synapses_15_N;
__device__ int32_t * brian::d_array_synapses_15_N;
const int brian::_num__array_synapses_15_N = 1;

int32_t * brian::_array_synapses_1_N;
int32_t * brian::dev_array_synapses_1_N;
__device__ int32_t * brian::d_array_synapses_1_N;
const int brian::_num__array_synapses_1_N = 1;

int32_t * brian::_array_synapses_2_N;
int32_t * brian::dev_array_synapses_2_N;
__device__ int32_t * brian::d_array_synapses_2_N;
const int brian::_num__array_synapses_2_N = 1;

int32_t * brian::_array_synapses_3_N;
int32_t * brian::dev_array_synapses_3_N;
__device__ int32_t * brian::d_array_synapses_3_N;
const int brian::_num__array_synapses_3_N = 1;

int32_t * brian::_array_synapses_4_N;
int32_t * brian::dev_array_synapses_4_N;
__device__ int32_t * brian::d_array_synapses_4_N;
const int brian::_num__array_synapses_4_N = 1;

int32_t * brian::_array_synapses_5_N;
int32_t * brian::dev_array_synapses_5_N;
__device__ int32_t * brian::d_array_synapses_5_N;
const int brian::_num__array_synapses_5_N = 1;

int32_t * brian::_array_synapses_6_N;
int32_t * brian::dev_array_synapses_6_N;
__device__ int32_t * brian::d_array_synapses_6_N;
const int brian::_num__array_synapses_6_N = 1;

int32_t * brian::_array_synapses_7_N;
int32_t * brian::dev_array_synapses_7_N;
__device__ int32_t * brian::d_array_synapses_7_N;
const int brian::_num__array_synapses_7_N = 1;

int32_t * brian::_array_synapses_8_N;
int32_t * brian::dev_array_synapses_8_N;
__device__ int32_t * brian::d_array_synapses_8_N;
const int brian::_num__array_synapses_8_N = 1;

int32_t * brian::_array_synapses_9_N;
int32_t * brian::dev_array_synapses_9_N;
__device__ int32_t * brian::d_array_synapses_9_N;
const int brian::_num__array_synapses_9_N = 1;

int32_t * brian::_array_synapses_N;
int32_t * brian::dev_array_synapses_N;
__device__ int32_t * brian::d_array_synapses_N;
const int brian::_num__array_synapses_N = 1;

int32_t * brian::_array_synapses_sources;
int32_t * brian::dev_array_synapses_sources;
__device__ int32_t * brian::d_array_synapses_sources;
const int brian::_num__array_synapses_sources = 27;

int32_t * brian::_array_synapses_sources_1;
int32_t * brian::dev_array_synapses_sources_1;
__device__ int32_t * brian::d_array_synapses_sources_1;
const int brian::_num__array_synapses_sources_1 = 243;

int32_t * brian::_array_synapses_targets;
int32_t * brian::dev_array_synapses_targets;
__device__ int32_t * brian::d_array_synapses_targets;
const int brian::_num__array_synapses_targets = 27;

int32_t * brian::_array_synapses_targets_1;
int32_t * brian::dev_array_synapses_targets_1;
__device__ int32_t * brian::d_array_synapses_targets_1;
const int brian::_num__array_synapses_targets_1 = 243;


//////////////// eventspaces ///////////////
// we dynamically create multiple eventspaces in no_or_const_delay_mode
// for initiating the first spikespace, we need a host pointer
// for choosing the right spikespace, we need a global index variable
int32_t * brian::_array_neurongroup_10__spikespace;
const int brian::_num__array_neurongroup_10__spikespace = 17;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_10__spikespace(1);
int brian::current_idx_array_neurongroup_10__spikespace = 0;
int32_t * brian::_array_neurongroup_11__spikespace;
const int brian::_num__array_neurongroup_11__spikespace = 121;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_11__spikespace(1);
int brian::current_idx_array_neurongroup_11__spikespace = 0;
int32_t * brian::_array_neurongroup_1__spikespace;
const int brian::_num__array_neurongroup_1__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_1__spikespace(1);
int brian::current_idx_array_neurongroup_1__spikespace = 0;
int32_t * brian::_array_neurongroup_2__spikespace;
const int brian::_num__array_neurongroup_2__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_2__spikespace(1);
int brian::current_idx_array_neurongroup_2__spikespace = 0;
int32_t * brian::_array_neurongroup_3__spikespace;
const int brian::_num__array_neurongroup_3__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_3__spikespace(1);
int brian::current_idx_array_neurongroup_3__spikespace = 0;
int32_t * brian::_array_neurongroup_4__spikespace;
const int brian::_num__array_neurongroup_4__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_4__spikespace(1);
int brian::current_idx_array_neurongroup_4__spikespace = 0;
int32_t * brian::_array_neurongroup_5__spikespace;
const int brian::_num__array_neurongroup_5__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_5__spikespace(1);
int brian::current_idx_array_neurongroup_5__spikespace = 0;
int32_t * brian::_array_neurongroup_6__spikespace;
const int brian::_num__array_neurongroup_6__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_6__spikespace(1);
int brian::current_idx_array_neurongroup_6__spikespace = 0;
int32_t * brian::_array_neurongroup_7__spikespace;
const int brian::_num__array_neurongroup_7__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_7__spikespace(1);
int brian::current_idx_array_neurongroup_7__spikespace = 0;
int32_t * brian::_array_neurongroup_8__spikespace;
const int brian::_num__array_neurongroup_8__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_8__spikespace(1);
int brian::current_idx_array_neurongroup_8__spikespace = 0;
int32_t * brian::_array_neurongroup_9__spikespace;
const int brian::_num__array_neurongroup_9__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup_9__spikespace(1);
int brian::current_idx_array_neurongroup_9__spikespace = 0;
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 3;
thrust::host_vector<int32_t*> brian::dev_array_neurongroup__spikespace(1);
int brian::current_idx_array_neurongroup__spikespace = 0;

//////////////// dynamic arrays 1d /////////
thrust::host_vector<int32_t> brian::_dynamic_array_spikemonitor_i;
thrust::device_vector<int32_t> brian::dev_dynamic_array_spikemonitor_i;
thrust::host_vector<double> brian::_dynamic_array_spikemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_spikemonitor_t;
thrust::host_vector<double> brian::_dynamic_array_statemonitor_t;
thrust::device_vector<double> brian::dev_dynamic_array_statemonitor_t;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_10__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_10__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_10__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_10__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_10_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_10_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_10_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_10_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_10_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_10_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_10_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_10_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_11__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_11__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_11__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_11__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_11_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_11_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_11_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_11_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_11_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_11_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_11_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_11_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_12__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_12__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_12__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_12__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_12_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_12_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_12_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_12_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_12_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_12_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_12_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_12_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_13__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_13__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_13__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_13__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_13_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_13_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_13_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_13_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_13_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_13_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_13_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_13_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_14__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_14__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_14__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_14__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_14_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_14_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_14_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_14_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_14_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_14_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_14_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_14_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_15__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_15__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_15__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_15__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_15_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_15_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_15_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_15_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_15_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_15_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_15_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_15_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_1_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_1_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_1_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_1_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_2_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_2_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_2_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_2_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_3_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_3_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_3_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_3_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_4_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_4_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_4_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_4_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_4_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_4_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_5_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_5_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_5_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_5_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_5_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_5_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_6_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_6_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_6_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_6_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_6_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_6_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_7__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_7__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_7__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_7__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_7_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_7_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_7_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_7_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_7_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_7_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_7_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_7_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_8__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_8__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_8__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_8__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_8_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_8_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_8_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_8_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_8_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_8_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_8_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_8_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_9__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_9__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_9__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_9__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_9_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_9_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_9_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_9_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_9_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_9_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_9_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_9_w;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_post;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses__synaptic_pre;
thrust::host_vector<double> brian::_dynamic_array_synapses_delay;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_delay;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_incoming;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_incoming;
thrust::host_vector<int32_t> brian::_dynamic_array_synapses_N_outgoing;
thrust::device_vector<int32_t> brian::dev_dynamic_array_synapses_N_outgoing;
thrust::host_vector<double> brian::_dynamic_array_synapses_w;
thrust::device_vector<double> brian::dev_dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_I;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_I;
thrust::device_vector<double*> brian::addresses_monitor__dynamic_array_statemonitor_v;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor_v;

/////////////// static arrays /////////////
int32_t * brian::_static_array__array_synapses_sources;
int32_t * brian::dev_static_array__array_synapses_sources;
__device__ int32_t * brian::d_static_array__array_synapses_sources;
const int brian::_num__static_array__array_synapses_sources = 27;
int32_t * brian::_static_array__array_synapses_sources_1;
int32_t * brian::dev_static_array__array_synapses_sources_1;
__device__ int32_t * brian::d_static_array__array_synapses_sources_1;
const int brian::_num__static_array__array_synapses_sources_1 = 243;
int32_t * brian::_static_array__array_synapses_targets;
int32_t * brian::dev_static_array__array_synapses_targets;
__device__ int32_t * brian::d_static_array__array_synapses_targets;
const int brian::_num__static_array__array_synapses_targets = 27;
int32_t * brian::_static_array__array_synapses_targets_1;
int32_t * brian::dev_static_array__array_synapses_targets_1;
__device__ int32_t * brian::d_static_array__array_synapses_targets_1;
const int brian::_num__static_array__array_synapses_targets_1 = 243;
double * brian::_timedarray_2_values;
double * brian::dev_timedarray_2_values;
__device__ double * brian::d_timedarray_2_values;
const int brian::_num__timedarray_2_values = 4800;
double * brian::_timedarray_values;
double * brian::dev_timedarray_values;
__device__ double * brian::d_timedarray_values;
const int brian::_num__timedarray_values = 9000;

//////////////// synapses /////////////////
// synapses
int32_t synapses_source_start_index;
int32_t synapses_source_stop_index;
bool brian::synapses_multiple_pre_post = false;
// synapses_pre
__device__ int* brian::synapses_pre_num_synapses_by_pre;
__device__ int* brian::synapses_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_pre_unique_delays;
__device__ int* brian::synapses_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_pre_global_bundle_id_start_by_pre;
int brian::synapses_pre_bundle_size_max = 0;
int brian::synapses_pre_bundle_size_min = 0;
double brian::synapses_pre_bundle_size_mean = 0;
double brian::synapses_pre_bundle_size_std = 0;
int brian::synapses_pre_max_size = 0;
__device__ int* brian::synapses_pre_num_unique_delays_by_pre;
int brian::synapses_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_pre_synapse_ids;
__device__ int* brian::synapses_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_pre;
int brian::synapses_pre_eventspace_idx = 0;
int brian::synapses_pre_delay;
bool brian::synapses_pre_scalar_delay;
// synapses_1
int32_t synapses_1_source_start_index;
int32_t synapses_1_source_stop_index;
bool brian::synapses_1_multiple_pre_post = false;
// synapses_1_pre
__device__ int* brian::synapses_1_pre_num_synapses_by_pre;
__device__ int* brian::synapses_1_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_1_pre_unique_delays;
__device__ int* brian::synapses_1_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_1_pre_global_bundle_id_start_by_pre;
int brian::synapses_1_pre_bundle_size_max = 0;
int brian::synapses_1_pre_bundle_size_min = 0;
double brian::synapses_1_pre_bundle_size_mean = 0;
double brian::synapses_1_pre_bundle_size_std = 0;
int brian::synapses_1_pre_max_size = 0;
__device__ int* brian::synapses_1_pre_num_unique_delays_by_pre;
int brian::synapses_1_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_1_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_1_pre_synapse_ids;
__device__ int* brian::synapses_1_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_1_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_1_pre;
int brian::synapses_1_pre_eventspace_idx = 0;
int brian::synapses_1_pre_delay;
bool brian::synapses_1_pre_scalar_delay;
// synapses_10
int32_t synapses_10_source_start_index;
int32_t synapses_10_source_stop_index;
bool brian::synapses_10_multiple_pre_post = false;
// synapses_10_pre
__device__ int* brian::synapses_10_pre_num_synapses_by_pre;
__device__ int* brian::synapses_10_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_10_pre_unique_delays;
__device__ int* brian::synapses_10_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_10_pre_global_bundle_id_start_by_pre;
int brian::synapses_10_pre_bundle_size_max = 0;
int brian::synapses_10_pre_bundle_size_min = 0;
double brian::synapses_10_pre_bundle_size_mean = 0;
double brian::synapses_10_pre_bundle_size_std = 0;
int brian::synapses_10_pre_max_size = 0;
__device__ int* brian::synapses_10_pre_num_unique_delays_by_pre;
int brian::synapses_10_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_10_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_10_pre_synapse_ids;
__device__ int* brian::synapses_10_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_10_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_10_pre;
int brian::synapses_10_pre_eventspace_idx = 0;
int brian::synapses_10_pre_delay;
bool brian::synapses_10_pre_scalar_delay;
// synapses_11
int32_t synapses_11_source_start_index;
int32_t synapses_11_source_stop_index;
bool brian::synapses_11_multiple_pre_post = false;
// synapses_11_pre
__device__ int* brian::synapses_11_pre_num_synapses_by_pre;
__device__ int* brian::synapses_11_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_11_pre_unique_delays;
__device__ int* brian::synapses_11_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_11_pre_global_bundle_id_start_by_pre;
int brian::synapses_11_pre_bundle_size_max = 0;
int brian::synapses_11_pre_bundle_size_min = 0;
double brian::synapses_11_pre_bundle_size_mean = 0;
double brian::synapses_11_pre_bundle_size_std = 0;
int brian::synapses_11_pre_max_size = 0;
__device__ int* brian::synapses_11_pre_num_unique_delays_by_pre;
int brian::synapses_11_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_11_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_11_pre_synapse_ids;
__device__ int* brian::synapses_11_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_11_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_11_pre;
int brian::synapses_11_pre_eventspace_idx = 0;
int brian::synapses_11_pre_delay;
bool brian::synapses_11_pre_scalar_delay;
// synapses_12
int32_t synapses_12_source_start_index;
int32_t synapses_12_source_stop_index;
bool brian::synapses_12_multiple_pre_post = false;
// synapses_12_pre
__device__ int* brian::synapses_12_pre_num_synapses_by_pre;
__device__ int* brian::synapses_12_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_12_pre_unique_delays;
__device__ int* brian::synapses_12_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_12_pre_global_bundle_id_start_by_pre;
int brian::synapses_12_pre_bundle_size_max = 0;
int brian::synapses_12_pre_bundle_size_min = 0;
double brian::synapses_12_pre_bundle_size_mean = 0;
double brian::synapses_12_pre_bundle_size_std = 0;
int brian::synapses_12_pre_max_size = 0;
__device__ int* brian::synapses_12_pre_num_unique_delays_by_pre;
int brian::synapses_12_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_12_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_12_pre_synapse_ids;
__device__ int* brian::synapses_12_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_12_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_12_pre;
int brian::synapses_12_pre_eventspace_idx = 0;
int brian::synapses_12_pre_delay;
bool brian::synapses_12_pre_scalar_delay;
// synapses_13
int32_t synapses_13_source_start_index;
int32_t synapses_13_source_stop_index;
bool brian::synapses_13_multiple_pre_post = false;
// synapses_13_pre
__device__ int* brian::synapses_13_pre_num_synapses_by_pre;
__device__ int* brian::synapses_13_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_13_pre_unique_delays;
__device__ int* brian::synapses_13_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_13_pre_global_bundle_id_start_by_pre;
int brian::synapses_13_pre_bundle_size_max = 0;
int brian::synapses_13_pre_bundle_size_min = 0;
double brian::synapses_13_pre_bundle_size_mean = 0;
double brian::synapses_13_pre_bundle_size_std = 0;
int brian::synapses_13_pre_max_size = 0;
__device__ int* brian::synapses_13_pre_num_unique_delays_by_pre;
int brian::synapses_13_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_13_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_13_pre_synapse_ids;
__device__ int* brian::synapses_13_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_13_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_13_pre;
int brian::synapses_13_pre_eventspace_idx = 0;
int brian::synapses_13_pre_delay;
bool brian::synapses_13_pre_scalar_delay;
// synapses_14
int32_t synapses_14_source_start_index;
int32_t synapses_14_source_stop_index;
bool brian::synapses_14_multiple_pre_post = false;
// synapses_14_pre
__device__ int* brian::synapses_14_pre_num_synapses_by_pre;
__device__ int* brian::synapses_14_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_14_pre_unique_delays;
__device__ int* brian::synapses_14_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_14_pre_global_bundle_id_start_by_pre;
int brian::synapses_14_pre_bundle_size_max = 0;
int brian::synapses_14_pre_bundle_size_min = 0;
double brian::synapses_14_pre_bundle_size_mean = 0;
double brian::synapses_14_pre_bundle_size_std = 0;
int brian::synapses_14_pre_max_size = 0;
__device__ int* brian::synapses_14_pre_num_unique_delays_by_pre;
int brian::synapses_14_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_14_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_14_pre_synapse_ids;
__device__ int* brian::synapses_14_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_14_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_14_pre;
int brian::synapses_14_pre_eventspace_idx = 0;
int brian::synapses_14_pre_delay;
bool brian::synapses_14_pre_scalar_delay;
// synapses_15
int32_t synapses_15_source_start_index;
int32_t synapses_15_source_stop_index;
bool brian::synapses_15_multiple_pre_post = false;
// synapses_15_pre
__device__ int* brian::synapses_15_pre_num_synapses_by_pre;
__device__ int* brian::synapses_15_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_15_pre_unique_delays;
__device__ int* brian::synapses_15_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_15_pre_global_bundle_id_start_by_pre;
int brian::synapses_15_pre_bundle_size_max = 0;
int brian::synapses_15_pre_bundle_size_min = 0;
double brian::synapses_15_pre_bundle_size_mean = 0;
double brian::synapses_15_pre_bundle_size_std = 0;
int brian::synapses_15_pre_max_size = 0;
__device__ int* brian::synapses_15_pre_num_unique_delays_by_pre;
int brian::synapses_15_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_15_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_15_pre_synapse_ids;
__device__ int* brian::synapses_15_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_15_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_15_pre;
int brian::synapses_15_pre_eventspace_idx = 0;
int brian::synapses_15_pre_delay;
bool brian::synapses_15_pre_scalar_delay;
// synapses_2
int32_t synapses_2_source_start_index;
int32_t synapses_2_source_stop_index;
bool brian::synapses_2_multiple_pre_post = false;
// synapses_2_pre
__device__ int* brian::synapses_2_pre_num_synapses_by_pre;
__device__ int* brian::synapses_2_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_2_pre_unique_delays;
__device__ int* brian::synapses_2_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_2_pre_global_bundle_id_start_by_pre;
int brian::synapses_2_pre_bundle_size_max = 0;
int brian::synapses_2_pre_bundle_size_min = 0;
double brian::synapses_2_pre_bundle_size_mean = 0;
double brian::synapses_2_pre_bundle_size_std = 0;
int brian::synapses_2_pre_max_size = 0;
__device__ int* brian::synapses_2_pre_num_unique_delays_by_pre;
int brian::synapses_2_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_2_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_2_pre_synapse_ids;
__device__ int* brian::synapses_2_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_2_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_2_pre;
int brian::synapses_2_pre_eventspace_idx = 0;
int brian::synapses_2_pre_delay;
bool brian::synapses_2_pre_scalar_delay;
// synapses_3
int32_t synapses_3_source_start_index;
int32_t synapses_3_source_stop_index;
bool brian::synapses_3_multiple_pre_post = false;
// synapses_3_pre
__device__ int* brian::synapses_3_pre_num_synapses_by_pre;
__device__ int* brian::synapses_3_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_3_pre_unique_delays;
__device__ int* brian::synapses_3_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_3_pre_global_bundle_id_start_by_pre;
int brian::synapses_3_pre_bundle_size_max = 0;
int brian::synapses_3_pre_bundle_size_min = 0;
double brian::synapses_3_pre_bundle_size_mean = 0;
double brian::synapses_3_pre_bundle_size_std = 0;
int brian::synapses_3_pre_max_size = 0;
__device__ int* brian::synapses_3_pre_num_unique_delays_by_pre;
int brian::synapses_3_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_3_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_3_pre_synapse_ids;
__device__ int* brian::synapses_3_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_3_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_3_pre;
int brian::synapses_3_pre_eventspace_idx = 0;
int brian::synapses_3_pre_delay;
bool brian::synapses_3_pre_scalar_delay;
// synapses_4
int32_t synapses_4_source_start_index;
int32_t synapses_4_source_stop_index;
bool brian::synapses_4_multiple_pre_post = false;
// synapses_4_pre
__device__ int* brian::synapses_4_pre_num_synapses_by_pre;
__device__ int* brian::synapses_4_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_4_pre_unique_delays;
__device__ int* brian::synapses_4_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_4_pre_global_bundle_id_start_by_pre;
int brian::synapses_4_pre_bundle_size_max = 0;
int brian::synapses_4_pre_bundle_size_min = 0;
double brian::synapses_4_pre_bundle_size_mean = 0;
double brian::synapses_4_pre_bundle_size_std = 0;
int brian::synapses_4_pre_max_size = 0;
__device__ int* brian::synapses_4_pre_num_unique_delays_by_pre;
int brian::synapses_4_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_4_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_4_pre_synapse_ids;
__device__ int* brian::synapses_4_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_4_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_4_pre;
int brian::synapses_4_pre_eventspace_idx = 0;
int brian::synapses_4_pre_delay;
bool brian::synapses_4_pre_scalar_delay;
// synapses_5
int32_t synapses_5_source_start_index;
int32_t synapses_5_source_stop_index;
bool brian::synapses_5_multiple_pre_post = false;
// synapses_5_pre
__device__ int* brian::synapses_5_pre_num_synapses_by_pre;
__device__ int* brian::synapses_5_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_5_pre_unique_delays;
__device__ int* brian::synapses_5_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_5_pre_global_bundle_id_start_by_pre;
int brian::synapses_5_pre_bundle_size_max = 0;
int brian::synapses_5_pre_bundle_size_min = 0;
double brian::synapses_5_pre_bundle_size_mean = 0;
double brian::synapses_5_pre_bundle_size_std = 0;
int brian::synapses_5_pre_max_size = 0;
__device__ int* brian::synapses_5_pre_num_unique_delays_by_pre;
int brian::synapses_5_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_5_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_5_pre_synapse_ids;
__device__ int* brian::synapses_5_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_5_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_5_pre;
int brian::synapses_5_pre_eventspace_idx = 0;
int brian::synapses_5_pre_delay;
bool brian::synapses_5_pre_scalar_delay;
// synapses_6
int32_t synapses_6_source_start_index;
int32_t synapses_6_source_stop_index;
bool brian::synapses_6_multiple_pre_post = false;
// synapses_6_pre
__device__ int* brian::synapses_6_pre_num_synapses_by_pre;
__device__ int* brian::synapses_6_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_6_pre_unique_delays;
__device__ int* brian::synapses_6_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_6_pre_global_bundle_id_start_by_pre;
int brian::synapses_6_pre_bundle_size_max = 0;
int brian::synapses_6_pre_bundle_size_min = 0;
double brian::synapses_6_pre_bundle_size_mean = 0;
double brian::synapses_6_pre_bundle_size_std = 0;
int brian::synapses_6_pre_max_size = 0;
__device__ int* brian::synapses_6_pre_num_unique_delays_by_pre;
int brian::synapses_6_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_6_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_6_pre_synapse_ids;
__device__ int* brian::synapses_6_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_6_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_6_pre;
int brian::synapses_6_pre_eventspace_idx = 0;
int brian::synapses_6_pre_delay;
bool brian::synapses_6_pre_scalar_delay;
// synapses_7
int32_t synapses_7_source_start_index;
int32_t synapses_7_source_stop_index;
bool brian::synapses_7_multiple_pre_post = false;
// synapses_7_pre
__device__ int* brian::synapses_7_pre_num_synapses_by_pre;
__device__ int* brian::synapses_7_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_7_pre_unique_delays;
__device__ int* brian::synapses_7_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_7_pre_global_bundle_id_start_by_pre;
int brian::synapses_7_pre_bundle_size_max = 0;
int brian::synapses_7_pre_bundle_size_min = 0;
double brian::synapses_7_pre_bundle_size_mean = 0;
double brian::synapses_7_pre_bundle_size_std = 0;
int brian::synapses_7_pre_max_size = 0;
__device__ int* brian::synapses_7_pre_num_unique_delays_by_pre;
int brian::synapses_7_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_7_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_7_pre_synapse_ids;
__device__ int* brian::synapses_7_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_7_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_7_pre;
int brian::synapses_7_pre_eventspace_idx = 0;
int brian::synapses_7_pre_delay;
bool brian::synapses_7_pre_scalar_delay;
// synapses_8
int32_t synapses_8_source_start_index;
int32_t synapses_8_source_stop_index;
bool brian::synapses_8_multiple_pre_post = false;
// synapses_8_pre
__device__ int* brian::synapses_8_pre_num_synapses_by_pre;
__device__ int* brian::synapses_8_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_8_pre_unique_delays;
__device__ int* brian::synapses_8_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_8_pre_global_bundle_id_start_by_pre;
int brian::synapses_8_pre_bundle_size_max = 0;
int brian::synapses_8_pre_bundle_size_min = 0;
double brian::synapses_8_pre_bundle_size_mean = 0;
double brian::synapses_8_pre_bundle_size_std = 0;
int brian::synapses_8_pre_max_size = 0;
__device__ int* brian::synapses_8_pre_num_unique_delays_by_pre;
int brian::synapses_8_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_8_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_8_pre_synapse_ids;
__device__ int* brian::synapses_8_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_8_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_8_pre;
int brian::synapses_8_pre_eventspace_idx = 0;
int brian::synapses_8_pre_delay;
bool brian::synapses_8_pre_scalar_delay;
// synapses_9
int32_t synapses_9_source_start_index;
int32_t synapses_9_source_stop_index;
bool brian::synapses_9_multiple_pre_post = false;
// synapses_9_pre
__device__ int* brian::synapses_9_pre_num_synapses_by_pre;
__device__ int* brian::synapses_9_pre_num_synapses_by_bundle;
__device__ int* brian::synapses_9_pre_unique_delays;
__device__ int* brian::synapses_9_pre_synapses_offset_by_bundle;
__device__ int* brian::synapses_9_pre_global_bundle_id_start_by_pre;
int brian::synapses_9_pre_bundle_size_max = 0;
int brian::synapses_9_pre_bundle_size_min = 0;
double brian::synapses_9_pre_bundle_size_mean = 0;
double brian::synapses_9_pre_bundle_size_std = 0;
int brian::synapses_9_pre_max_size = 0;
__device__ int* brian::synapses_9_pre_num_unique_delays_by_pre;
int brian::synapses_9_pre_max_num_unique_delays = 0;
__device__ int32_t** brian::synapses_9_pre_synapse_ids_by_pre;
__device__ int32_t* brian::synapses_9_pre_synapse_ids;
__device__ int* brian::synapses_9_pre_unique_delay_start_idcs;
__device__ int* brian::synapses_9_pre_unique_delays_offset_by_pre;
__device__ SynapticPathway brian::synapses_9_pre;
int brian::synapses_9_pre_eventspace_idx = 0;
int brian::synapses_9_pre_delay;
bool brian::synapses_9_pre_scalar_delay;

int brian::num_parallel_blocks;
int brian::max_threads_per_block;
int brian::max_threads_per_sm;
int brian::max_shared_mem_size;
int brian::num_threads_per_warp;

__global__ void synapses_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_1_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_1_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_10_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_10_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_11_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_11_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_12_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_12_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_13_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_13_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_14_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_14_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_15_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_15_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_2_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_2_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_3_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_3_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_4_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_4_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_5_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_5_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_6_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_6_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_7_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_7_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_8_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_8_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}
__global__ void synapses_9_pre_init(
                int32_t* sources,
                int32_t* targets,
                double dt,
                int32_t source_start,
                int32_t source_stop
        )
{
    using namespace brian;

    synapses_9_pre.init(
            sources,
            targets,
            dt,
            // TODO: called source here, spikes in SynapticPathway (use same name)
            source_start,
            source_stop);
}

// Profiling information for each code object

//////////////random numbers//////////////////
curandGenerator_t brian::curand_generator;
__device__ unsigned long long* brian::d_curand_seed;
unsigned long long* brian::dev_curand_seed;
// dev_{co.name}_{rng_type}_allocator
//      pointer to start of generated random numbers array
//      at each generation cycle this array is refilled
// dev_{co.name}_{rng_type}
//      pointer moving through generated random number array
//      until it is regenerated at the next generation cycle
curandState* brian::dev_curand_states;
__device__ curandState* brian::d_curand_states;
RandomNumberBuffer brian::random_number_buffer;

void _init_arrays()
{
    using namespace brian;

    std::clock_t start_timer = std::clock();

    CUDA_CHECK_MEMORY();
    size_t used_device_memory_start = used_device_memory;

    cudaDeviceProp props;
    CUDA_SAFE_CALL(
            cudaGetDeviceProperties(&props, 0)
            );

    num_parallel_blocks = 1;
    max_threads_per_block = props.maxThreadsPerBlock;
    max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    max_shared_mem_size = props.sharedMemPerBlock;
    num_threads_per_warp = props.warpSize;

    // Random seeds might be overwritten in main.cu
    unsigned long long seed = time(0);

    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_curand_seed,
                sizeof(unsigned long long))
            );

    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_curand_seed, &dev_curand_seed,
                sizeof(unsigned long long*))
            );

    CUDA_SAFE_CALL(
            curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT)
            );


    // this sets seed for host and device api RNG
    random_number_buffer.set_seed(seed);

    synapses_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            120
            );
    CUDA_CHECK_ERROR("synapses_pre_init");
    synapses_1_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_1__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_1_pre_init");
    synapses_10_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_10__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_10__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_10_pre_init");
    synapses_11_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_11__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_11__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_11_pre_init");
    synapses_12_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_12__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_12__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_12_pre_init");
    synapses_13_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_13__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_13__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_13_pre_init");
    synapses_14_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_14__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_14__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            16
            );
    CUDA_CHECK_ERROR("synapses_14_pre_init");
    synapses_15_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_15__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_15__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_15_pre_init");
    synapses_2_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_2__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_2_pre_init");
    synapses_3_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_3__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_3_pre_init");
    synapses_4_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_4__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_4_pre_init");
    synapses_5_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_5__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_5_pre_init");
    synapses_6_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_6__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_6__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_6_pre_init");
    synapses_7_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_7__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_7__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_7_pre_init");
    synapses_8_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_8__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_8__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_8_pre_init");
    synapses_9_pre_init<<<1,1>>>(
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_9__synaptic_pre[0]),
            thrust::raw_pointer_cast(&dev_dynamic_array_synapses_9__synaptic_post[0]),
            0,  //was dt, maybe irrelevant?
            0,
            2
            );
    CUDA_CHECK_ERROR("synapses_9_pre_init");

    // Arrays initialized to 0
            _array_defaultclock_dt = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_dt, _array_defaultclock_dt, sizeof(double)*_num__array_defaultclock_dt, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_t = new double[1];
            for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_t, _array_defaultclock_t, sizeof(double)*_num__array_defaultclock_t, cudaMemcpyHostToDevice)
                    );
            _array_defaultclock_timestep = new int64_t[1];
            for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_defaultclock_timestep, _array_defaultclock_timestep, sizeof(int64_t)*_num__array_defaultclock_timestep, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_A = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_A, sizeof(double)*_num__array_neurongroup_10_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_A, _array_neurongroup_10_A, sizeof(double)*_num__array_neurongroup_10_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_batch_sum_X = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_batch_sum_X, sizeof(double)*_num__array_neurongroup_10_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_batch_sum_X, _array_neurongroup_10_batch_sum_X, sizeof(double)*_num__array_neurongroup_10_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_i = new int32_t[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_i, sizeof(int32_t)*_num__array_neurongroup_10_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_i, _array_neurongroup_10_i, sizeof(int32_t)*_num__array_neurongroup_10_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_Iexc = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_Iexc, sizeof(double)*_num__array_neurongroup_10_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_Iexc, _array_neurongroup_10_Iexc, sizeof(double)*_num__array_neurongroup_10_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_Iinh = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_Iinh, sizeof(double)*_num__array_neurongroup_10_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_Iinh, _array_neurongroup_10_Iinh, sizeof(double)*_num__array_neurongroup_10_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_running_sum_X = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_running_sum_X, sizeof(double)*_num__array_neurongroup_10_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_running_sum_X, _array_neurongroup_10_running_sum_X, sizeof(double)*_num__array_neurongroup_10_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_v = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_v, sizeof(double)*_num__array_neurongroup_10_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_v, _array_neurongroup_10_v, sizeof(double)*_num__array_neurongroup_10_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_X = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_X, sizeof(double)*_num__array_neurongroup_10_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_X, _array_neurongroup_10_X, sizeof(double)*_num__array_neurongroup_10_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_10_Y = new double[16];
            for(int i=0; i<16; i++) _array_neurongroup_10_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_10_Y, sizeof(double)*_num__array_neurongroup_10_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_10_Y, _array_neurongroup_10_Y, sizeof(double)*_num__array_neurongroup_10_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_A = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_A, sizeof(double)*_num__array_neurongroup_11_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_A, _array_neurongroup_11_A, sizeof(double)*_num__array_neurongroup_11_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_i = new int32_t[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_i, sizeof(int32_t)*_num__array_neurongroup_11_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_i, _array_neurongroup_11_i, sizeof(int32_t)*_num__array_neurongroup_11_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_Iexc = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_Iexc, sizeof(double)*_num__array_neurongroup_11_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_Iexc, _array_neurongroup_11_Iexc, sizeof(double)*_num__array_neurongroup_11_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_Iexc2 = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_Iexc2[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_Iexc2, sizeof(double)*_num__array_neurongroup_11_Iexc2)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_Iexc2, _array_neurongroup_11_Iexc2, sizeof(double)*_num__array_neurongroup_11_Iexc2, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_indices = new int32_t[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_indices, sizeof(int32_t)*_num__array_neurongroup_11_indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_indices, _array_neurongroup_11_indices, sizeof(int32_t)*_num__array_neurongroup_11_indices, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_v = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_v, sizeof(double)*_num__array_neurongroup_11_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_v, _array_neurongroup_11_v, sizeof(double)*_num__array_neurongroup_11_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_X = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_X, sizeof(double)*_num__array_neurongroup_11_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_X, _array_neurongroup_11_X, sizeof(double)*_num__array_neurongroup_11_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_11_Y = new double[120];
            for(int i=0; i<120; i++) _array_neurongroup_11_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_11_Y, sizeof(double)*_num__array_neurongroup_11_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_11_Y, _array_neurongroup_11_Y, sizeof(double)*_num__array_neurongroup_11_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_A, sizeof(double)*_num__array_neurongroup_1_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_A, _array_neurongroup_1_A, sizeof(double)*_num__array_neurongroup_1_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_batch_sum_X, sizeof(double)*_num__array_neurongroup_1_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_batch_sum_X, _array_neurongroup_1_batch_sum_X, sizeof(double)*_num__array_neurongroup_1_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_Iexc, sizeof(double)*_num__array_neurongroup_1_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_Iexc, _array_neurongroup_1_Iexc, sizeof(double)*_num__array_neurongroup_1_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_Iinh, sizeof(double)*_num__array_neurongroup_1_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_Iinh, _array_neurongroup_1_Iinh, sizeof(double)*_num__array_neurongroup_1_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_running_sum_X, sizeof(double)*_num__array_neurongroup_1_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_running_sum_X, _array_neurongroup_1_running_sum_X, sizeof(double)*_num__array_neurongroup_1_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_v, sizeof(double)*_num__array_neurongroup_1_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_v, _array_neurongroup_1_v, sizeof(double)*_num__array_neurongroup_1_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_X, sizeof(double)*_num__array_neurongroup_1_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_X, _array_neurongroup_1_X, sizeof(double)*_num__array_neurongroup_1_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_1_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_1_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_1_Y, sizeof(double)*_num__array_neurongroup_1_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_1_Y, _array_neurongroup_1_Y, sizeof(double)*_num__array_neurongroup_1_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_A, sizeof(double)*_num__array_neurongroup_2_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_A, _array_neurongroup_2_A, sizeof(double)*_num__array_neurongroup_2_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_batch_sum_X, sizeof(double)*_num__array_neurongroup_2_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_batch_sum_X, _array_neurongroup_2_batch_sum_X, sizeof(double)*_num__array_neurongroup_2_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_Iexc, sizeof(double)*_num__array_neurongroup_2_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_Iexc, _array_neurongroup_2_Iexc, sizeof(double)*_num__array_neurongroup_2_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_Iinh, sizeof(double)*_num__array_neurongroup_2_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_Iinh, _array_neurongroup_2_Iinh, sizeof(double)*_num__array_neurongroup_2_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_running_sum_X, sizeof(double)*_num__array_neurongroup_2_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_running_sum_X, _array_neurongroup_2_running_sum_X, sizeof(double)*_num__array_neurongroup_2_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_v, _array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_X, sizeof(double)*_num__array_neurongroup_2_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_X, _array_neurongroup_2_X, sizeof(double)*_num__array_neurongroup_2_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_2_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_2_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_2_Y, sizeof(double)*_num__array_neurongroup_2_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_2_Y, _array_neurongroup_2_Y, sizeof(double)*_num__array_neurongroup_2_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_A, sizeof(double)*_num__array_neurongroup_3_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_A, _array_neurongroup_3_A, sizeof(double)*_num__array_neurongroup_3_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_batch_sum_X, sizeof(double)*_num__array_neurongroup_3_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_batch_sum_X, _array_neurongroup_3_batch_sum_X, sizeof(double)*_num__array_neurongroup_3_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_i, _array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_Iexc, sizeof(double)*_num__array_neurongroup_3_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_Iexc, _array_neurongroup_3_Iexc, sizeof(double)*_num__array_neurongroup_3_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_Iinh, sizeof(double)*_num__array_neurongroup_3_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_Iinh, _array_neurongroup_3_Iinh, sizeof(double)*_num__array_neurongroup_3_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_running_sum_X, sizeof(double)*_num__array_neurongroup_3_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_running_sum_X, _array_neurongroup_3_running_sum_X, sizeof(double)*_num__array_neurongroup_3_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_v, sizeof(double)*_num__array_neurongroup_3_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_v, _array_neurongroup_3_v, sizeof(double)*_num__array_neurongroup_3_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_X, sizeof(double)*_num__array_neurongroup_3_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_X, _array_neurongroup_3_X, sizeof(double)*_num__array_neurongroup_3_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_3_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_3_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_3_Y, sizeof(double)*_num__array_neurongroup_3_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_3_Y, _array_neurongroup_3_Y, sizeof(double)*_num__array_neurongroup_3_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_A, sizeof(double)*_num__array_neurongroup_4_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_A, _array_neurongroup_4_A, sizeof(double)*_num__array_neurongroup_4_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_batch_sum_X, sizeof(double)*_num__array_neurongroup_4_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_batch_sum_X, _array_neurongroup_4_batch_sum_X, sizeof(double)*_num__array_neurongroup_4_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_i, sizeof(int32_t)*_num__array_neurongroup_4_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_i, _array_neurongroup_4_i, sizeof(int32_t)*_num__array_neurongroup_4_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_Iexc, sizeof(double)*_num__array_neurongroup_4_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_Iexc, _array_neurongroup_4_Iexc, sizeof(double)*_num__array_neurongroup_4_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_Iinh, sizeof(double)*_num__array_neurongroup_4_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_Iinh, _array_neurongroup_4_Iinh, sizeof(double)*_num__array_neurongroup_4_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_running_sum_X, sizeof(double)*_num__array_neurongroup_4_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_running_sum_X, _array_neurongroup_4_running_sum_X, sizeof(double)*_num__array_neurongroup_4_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_v, sizeof(double)*_num__array_neurongroup_4_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_v, _array_neurongroup_4_v, sizeof(double)*_num__array_neurongroup_4_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_X, sizeof(double)*_num__array_neurongroup_4_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_X, _array_neurongroup_4_X, sizeof(double)*_num__array_neurongroup_4_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_4_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_4_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_4_Y, sizeof(double)*_num__array_neurongroup_4_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_4_Y, _array_neurongroup_4_Y, sizeof(double)*_num__array_neurongroup_4_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_A, sizeof(double)*_num__array_neurongroup_5_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_A, _array_neurongroup_5_A, sizeof(double)*_num__array_neurongroup_5_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_batch_sum_X, sizeof(double)*_num__array_neurongroup_5_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_batch_sum_X, _array_neurongroup_5_batch_sum_X, sizeof(double)*_num__array_neurongroup_5_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_i, sizeof(int32_t)*_num__array_neurongroup_5_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_i, _array_neurongroup_5_i, sizeof(int32_t)*_num__array_neurongroup_5_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_Iexc, sizeof(double)*_num__array_neurongroup_5_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_Iexc, _array_neurongroup_5_Iexc, sizeof(double)*_num__array_neurongroup_5_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_Iinh, sizeof(double)*_num__array_neurongroup_5_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_Iinh, _array_neurongroup_5_Iinh, sizeof(double)*_num__array_neurongroup_5_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_running_sum_X, sizeof(double)*_num__array_neurongroup_5_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_running_sum_X, _array_neurongroup_5_running_sum_X, sizeof(double)*_num__array_neurongroup_5_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_v, sizeof(double)*_num__array_neurongroup_5_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_v, _array_neurongroup_5_v, sizeof(double)*_num__array_neurongroup_5_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_X, sizeof(double)*_num__array_neurongroup_5_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_X, _array_neurongroup_5_X, sizeof(double)*_num__array_neurongroup_5_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_5_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_5_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_5_Y, sizeof(double)*_num__array_neurongroup_5_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_5_Y, _array_neurongroup_5_Y, sizeof(double)*_num__array_neurongroup_5_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_A, sizeof(double)*_num__array_neurongroup_6_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_A, _array_neurongroup_6_A, sizeof(double)*_num__array_neurongroup_6_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_batch_sum_X, sizeof(double)*_num__array_neurongroup_6_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_batch_sum_X, _array_neurongroup_6_batch_sum_X, sizeof(double)*_num__array_neurongroup_6_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_i, sizeof(int32_t)*_num__array_neurongroup_6_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_i, _array_neurongroup_6_i, sizeof(int32_t)*_num__array_neurongroup_6_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_Iexc, sizeof(double)*_num__array_neurongroup_6_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_Iexc, _array_neurongroup_6_Iexc, sizeof(double)*_num__array_neurongroup_6_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_Iinh, sizeof(double)*_num__array_neurongroup_6_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_Iinh, _array_neurongroup_6_Iinh, sizeof(double)*_num__array_neurongroup_6_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_running_sum_X, sizeof(double)*_num__array_neurongroup_6_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_running_sum_X, _array_neurongroup_6_running_sum_X, sizeof(double)*_num__array_neurongroup_6_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_v, sizeof(double)*_num__array_neurongroup_6_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_v, _array_neurongroup_6_v, sizeof(double)*_num__array_neurongroup_6_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_X, sizeof(double)*_num__array_neurongroup_6_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_X, _array_neurongroup_6_X, sizeof(double)*_num__array_neurongroup_6_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_6_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_6_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_6_Y, sizeof(double)*_num__array_neurongroup_6_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_6_Y, _array_neurongroup_6_Y, sizeof(double)*_num__array_neurongroup_6_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_A, sizeof(double)*_num__array_neurongroup_7_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_A, _array_neurongroup_7_A, sizeof(double)*_num__array_neurongroup_7_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_batch_sum_X, sizeof(double)*_num__array_neurongroup_7_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_batch_sum_X, _array_neurongroup_7_batch_sum_X, sizeof(double)*_num__array_neurongroup_7_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_i, sizeof(int32_t)*_num__array_neurongroup_7_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_i, _array_neurongroup_7_i, sizeof(int32_t)*_num__array_neurongroup_7_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_Iexc, sizeof(double)*_num__array_neurongroup_7_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_Iexc, _array_neurongroup_7_Iexc, sizeof(double)*_num__array_neurongroup_7_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_Iinh, sizeof(double)*_num__array_neurongroup_7_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_Iinh, _array_neurongroup_7_Iinh, sizeof(double)*_num__array_neurongroup_7_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_running_sum_X, sizeof(double)*_num__array_neurongroup_7_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_running_sum_X, _array_neurongroup_7_running_sum_X, sizeof(double)*_num__array_neurongroup_7_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_v, sizeof(double)*_num__array_neurongroup_7_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_v, _array_neurongroup_7_v, sizeof(double)*_num__array_neurongroup_7_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_X, sizeof(double)*_num__array_neurongroup_7_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_X, _array_neurongroup_7_X, sizeof(double)*_num__array_neurongroup_7_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_7_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_7_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_7_Y, sizeof(double)*_num__array_neurongroup_7_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_7_Y, _array_neurongroup_7_Y, sizeof(double)*_num__array_neurongroup_7_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_A, sizeof(double)*_num__array_neurongroup_8_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_A, _array_neurongroup_8_A, sizeof(double)*_num__array_neurongroup_8_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_batch_sum_X, sizeof(double)*_num__array_neurongroup_8_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_batch_sum_X, _array_neurongroup_8_batch_sum_X, sizeof(double)*_num__array_neurongroup_8_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_i, sizeof(int32_t)*_num__array_neurongroup_8_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_i, _array_neurongroup_8_i, sizeof(int32_t)*_num__array_neurongroup_8_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_Iexc, sizeof(double)*_num__array_neurongroup_8_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_Iexc, _array_neurongroup_8_Iexc, sizeof(double)*_num__array_neurongroup_8_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_Iinh, sizeof(double)*_num__array_neurongroup_8_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_Iinh, _array_neurongroup_8_Iinh, sizeof(double)*_num__array_neurongroup_8_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_running_sum_X, sizeof(double)*_num__array_neurongroup_8_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_running_sum_X, _array_neurongroup_8_running_sum_X, sizeof(double)*_num__array_neurongroup_8_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_v, sizeof(double)*_num__array_neurongroup_8_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_v, _array_neurongroup_8_v, sizeof(double)*_num__array_neurongroup_8_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_X, sizeof(double)*_num__array_neurongroup_8_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_X, _array_neurongroup_8_X, sizeof(double)*_num__array_neurongroup_8_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_8_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_8_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_8_Y, sizeof(double)*_num__array_neurongroup_8_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_8_Y, _array_neurongroup_8_Y, sizeof(double)*_num__array_neurongroup_8_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_A, sizeof(double)*_num__array_neurongroup_9_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_A, _array_neurongroup_9_A, sizeof(double)*_num__array_neurongroup_9_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_batch_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_batch_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_batch_sum_X, sizeof(double)*_num__array_neurongroup_9_batch_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_batch_sum_X, _array_neurongroup_9_batch_sum_X, sizeof(double)*_num__array_neurongroup_9_batch_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_i, sizeof(int32_t)*_num__array_neurongroup_9_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_i, _array_neurongroup_9_i, sizeof(int32_t)*_num__array_neurongroup_9_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_Iexc = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_Iexc[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_Iexc, sizeof(double)*_num__array_neurongroup_9_Iexc)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_Iexc, _array_neurongroup_9_Iexc, sizeof(double)*_num__array_neurongroup_9_Iexc, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_Iinh = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_Iinh[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_Iinh, sizeof(double)*_num__array_neurongroup_9_Iinh)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_Iinh, _array_neurongroup_9_Iinh, sizeof(double)*_num__array_neurongroup_9_Iinh, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_running_sum_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_running_sum_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_running_sum_X, sizeof(double)*_num__array_neurongroup_9_running_sum_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_running_sum_X, _array_neurongroup_9_running_sum_X, sizeof(double)*_num__array_neurongroup_9_running_sum_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_v, sizeof(double)*_num__array_neurongroup_9_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_v, _array_neurongroup_9_v, sizeof(double)*_num__array_neurongroup_9_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_X, sizeof(double)*_num__array_neurongroup_9_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_X, _array_neurongroup_9_X, sizeof(double)*_num__array_neurongroup_9_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_9_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_9_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_9_Y, sizeof(double)*_num__array_neurongroup_9_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_9_Y, _array_neurongroup_9_Y, sizeof(double)*_num__array_neurongroup_9_Y, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_A = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_A[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_A, sizeof(double)*_num__array_neurongroup_A)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_A, _array_neurongroup_A, sizeof(double)*_num__array_neurongroup_A, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_i = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_i[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_indices = new int32_t[2];
            for(int i=0; i<2; i++) _array_neurongroup_indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_indices, sizeof(int32_t)*_num__array_neurongroup_indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_indices, _array_neurongroup_indices, sizeof(int32_t)*_num__array_neurongroup_indices, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_v = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_v[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_X = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_X[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_X, sizeof(double)*_num__array_neurongroup_X)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_X, _array_neurongroup_X, sizeof(double)*_num__array_neurongroup_X, cudaMemcpyHostToDevice)
                    );
            _array_neurongroup_Y = new double[2];
            for(int i=0; i<2; i++) _array_neurongroup_Y[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_neurongroup_Y, sizeof(double)*_num__array_neurongroup_Y)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_neurongroup_Y, _array_neurongroup_Y, sizeof(double)*_num__array_neurongroup_Y, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor__source_idx = new int32_t[2];
            for(int i=0; i<2; i++) _array_spikemonitor__source_idx[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_count = new int32_t[2];
            for(int i=0; i<2; i++) _array_spikemonitor_count[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_count, _array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyHostToDevice)
                    );
            _array_spikemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_spikemonitor_N, _array_spikemonitor_N, sizeof(int32_t)*_num__array_spikemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor__indices = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor__indices[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor__indices, _array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyHostToDevice)
                    );
            _array_statemonitor_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_statemonitor_N, _array_statemonitor_N, sizeof(int32_t)*_num__array_statemonitor_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_10_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_10_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_10_N, sizeof(int32_t)*_num__array_synapses_10_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_10_N, _array_synapses_10_N, sizeof(int32_t)*_num__array_synapses_10_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_11_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_11_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_11_N, sizeof(int32_t)*_num__array_synapses_11_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_11_N, _array_synapses_11_N, sizeof(int32_t)*_num__array_synapses_11_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_12_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_12_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_12_N, sizeof(int32_t)*_num__array_synapses_12_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_12_N, _array_synapses_12_N, sizeof(int32_t)*_num__array_synapses_12_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_13_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_13_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_13_N, sizeof(int32_t)*_num__array_synapses_13_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_13_N, _array_synapses_13_N, sizeof(int32_t)*_num__array_synapses_13_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_14_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_14_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_14_N, sizeof(int32_t)*_num__array_synapses_14_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_14_N, _array_synapses_14_N, sizeof(int32_t)*_num__array_synapses_14_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_15_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_15_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_15_N, sizeof(int32_t)*_num__array_synapses_15_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_15_N, _array_synapses_15_N, sizeof(int32_t)*_num__array_synapses_15_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_1_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_1_N, _array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_2_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_2_N, _array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_3_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_3_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_3_N, _array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_4_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_4_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_4_N, _array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_5_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_5_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_5_N, _array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_6_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_6_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_6_N, _array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_7_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_7_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_7_N, sizeof(int32_t)*_num__array_synapses_7_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_7_N, _array_synapses_7_N, sizeof(int32_t)*_num__array_synapses_7_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_8_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_8_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_8_N, sizeof(int32_t)*_num__array_synapses_8_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_8_N, _array_synapses_8_N, sizeof(int32_t)*_num__array_synapses_8_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_9_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_9_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_9_N, sizeof(int32_t)*_num__array_synapses_9_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_9_N, _array_synapses_9_N, sizeof(int32_t)*_num__array_synapses_9_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_N = new int32_t[1];
            for(int i=0; i<1; i++) _array_synapses_N[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_N, _array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyHostToDevice)
                    );
            _array_synapses_sources = new int32_t[27];
            for(int i=0; i<27; i++) _array_synapses_sources[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_sources, _array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources, cudaMemcpyHostToDevice)
                    );
            _array_synapses_sources_1 = new int32_t[243];
            for(int i=0; i<243; i++) _array_synapses_sources_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_sources_1, sizeof(int32_t)*_num__array_synapses_sources_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_sources_1, _array_synapses_sources_1, sizeof(int32_t)*_num__array_synapses_sources_1, cudaMemcpyHostToDevice)
                    );
            _array_synapses_targets = new int32_t[27];
            for(int i=0; i<27; i++) _array_synapses_targets[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_targets, _array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets, cudaMemcpyHostToDevice)
                    );
            _array_synapses_targets_1 = new int32_t[243];
            for(int i=0; i<243; i++) _array_synapses_targets_1[i] = 0;
            CUDA_SAFE_CALL(
                    cudaMalloc((void**)&dev_array_synapses_targets_1, sizeof(int32_t)*_num__array_synapses_targets_1)
                    );
            CUDA_SAFE_CALL(
                    cudaMemcpy(dev_array_synapses_targets_1, _array_synapses_targets_1, sizeof(int32_t)*_num__array_synapses_targets_1, cudaMemcpyHostToDevice)
                    );

    // Arrays initialized to an "arange"
    _array_neurongroup_10_i = new int32_t[16];
    for(int i=0; i<16; i++) _array_neurongroup_10_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_10_i, sizeof(int32_t)*_num__array_neurongroup_10_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_10_i, _array_neurongroup_10_i, sizeof(int32_t)*_num__array_neurongroup_10_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_11_i = new int32_t[120];
    for(int i=0; i<120; i++) _array_neurongroup_11_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_11_i, sizeof(int32_t)*_num__array_neurongroup_11_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_11_i, _array_neurongroup_11_i, sizeof(int32_t)*_num__array_neurongroup_11_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_1_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_1_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_1_i, _array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_2_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_2_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_2_i, _array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_3_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_3_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_3_i, _array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_4_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_4_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_4_i, sizeof(int32_t)*_num__array_neurongroup_4_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_4_i, _array_neurongroup_4_i, sizeof(int32_t)*_num__array_neurongroup_4_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_5_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_5_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_5_i, sizeof(int32_t)*_num__array_neurongroup_5_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_5_i, _array_neurongroup_5_i, sizeof(int32_t)*_num__array_neurongroup_5_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_6_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_6_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_6_i, sizeof(int32_t)*_num__array_neurongroup_6_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_6_i, _array_neurongroup_6_i, sizeof(int32_t)*_num__array_neurongroup_6_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_7_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_7_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_7_i, sizeof(int32_t)*_num__array_neurongroup_7_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_7_i, _array_neurongroup_7_i, sizeof(int32_t)*_num__array_neurongroup_7_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_8_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_8_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_8_i, sizeof(int32_t)*_num__array_neurongroup_8_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_8_i, _array_neurongroup_8_i, sizeof(int32_t)*_num__array_neurongroup_8_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_9_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_9_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_9_i, sizeof(int32_t)*_num__array_neurongroup_9_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_9_i, _array_neurongroup_9_i, sizeof(int32_t)*_num__array_neurongroup_9_i, cudaMemcpyHostToDevice)
            );
    _array_neurongroup_i = new int32_t[2];
    for(int i=0; i<2; i++) _array_neurongroup_i[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice)
            );
    _array_spikemonitor__source_idx = new int32_t[2];
    for(int i=0; i<2; i++) _array_spikemonitor__source_idx[i] = 0 + i;
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx)
            );

    CUDA_SAFE_CALL(
            cudaMemcpy(dev_array_spikemonitor__source_idx, _array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyHostToDevice)
            );

    // static arrays
    _static_array__array_synapses_sources = new int32_t[27];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_sources, sizeof(int32_t)*27)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_sources, &dev_static_array__array_synapses_sources, sizeof(int32_t*))
            );
    _static_array__array_synapses_sources_1 = new int32_t[243];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_sources_1, sizeof(int32_t)*243)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_sources_1, &dev_static_array__array_synapses_sources_1, sizeof(int32_t*))
            );
    _static_array__array_synapses_targets = new int32_t[27];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_targets, sizeof(int32_t)*27)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_targets, &dev_static_array__array_synapses_targets, sizeof(int32_t*))
            );
    _static_array__array_synapses_targets_1 = new int32_t[243];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_static_array__array_synapses_targets_1, sizeof(int32_t)*243)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_static_array__array_synapses_targets_1, &dev_static_array__array_synapses_targets_1, sizeof(int32_t*))
            );
    _timedarray_2_values = new double[4800];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_2_values, sizeof(double)*4800)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_2_values, &dev_timedarray_2_values, sizeof(double*))
            );
    _timedarray_values = new double[9000];
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_timedarray_values, sizeof(double)*9000)
            );
    CUDA_SAFE_CALL(
            cudaMemcpyToSymbol(d_timedarray_values, &dev_timedarray_values, sizeof(double*))
            );

    _dynamic_array_statemonitor_I = new thrust::device_vector<double>[_num__array_statemonitor__indices];
    _dynamic_array_statemonitor_v = new thrust::device_vector<double>[_num__array_statemonitor__indices];

    // eventspace_arrays
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_10__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_10__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_10__spikespace = new int32_t[17];
    for (int i=0; i<17-1; i++)
    {
        _array_neurongroup_10__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_10__spikespace[17 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_10__spikespace[0],
            _array_neurongroup_10__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_10__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_11__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_11__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_11__spikespace = new int32_t[121];
    for (int i=0; i<121-1; i++)
    {
        _array_neurongroup_11__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_11__spikespace[121 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_11__spikespace[0],
            _array_neurongroup_11__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_11__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_1__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_1__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_1__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_1__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_1__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_1__spikespace[0],
            _array_neurongroup_1__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_1__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_2__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_2__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_2__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_2__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_2__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_2__spikespace[0],
            _array_neurongroup_2__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_2__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_3__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_3__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_3__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_3__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_3__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_3__spikespace[0],
            _array_neurongroup_3__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_3__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_4__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_4__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_4__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_4__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_4__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_4__spikespace[0],
            _array_neurongroup_4__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_4__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_5__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_5__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_5__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_5__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_5__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_5__spikespace[0],
            _array_neurongroup_5__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_5__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_6__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_6__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_6__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_6__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_6__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_6__spikespace[0],
            _array_neurongroup_6__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_6__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_7__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_7__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_7__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_7__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_7__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_7__spikespace[0],
            _array_neurongroup_7__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_7__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_8__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_8__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_8__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_8__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_8__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_8__spikespace[0],
            _array_neurongroup_8__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_8__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup_9__spikespace[0], sizeof(int32_t)*_num__array_neurongroup_9__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup_9__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup_9__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup_9__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup_9__spikespace[0],
            _array_neurongroup_9__spikespace,
            sizeof(int32_t) * _num__array_neurongroup_9__spikespace,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_SAFE_CALL(
            cudaMalloc((void**)&dev_array_neurongroup__spikespace[0], sizeof(int32_t)*_num__array_neurongroup__spikespace)
            );
    // initialize eventspace with -1
    _array_neurongroup__spikespace = new int32_t[3];
    for (int i=0; i<3-1; i++)
    {
        _array_neurongroup__spikespace[i] = -1;
    }
    // initialize eventspace counter with 0
    _array_neurongroup__spikespace[3 - 1] = 0;
    CUDA_SAFE_CALL(
        cudaMemcpy(
            dev_array_neurongroup__spikespace[0],
            _array_neurongroup__spikespace,
            sizeof(int32_t) * _num__array_neurongroup__spikespace,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK_MEMORY();
    const double to_MB = 1.0 / (1024.0 * 1024.0);
    double tot_memory_MB = (used_device_memory - used_device_memory_start) * to_MB;
    double time_passed = (double)(std::clock() - start_timer) / CLOCKS_PER_SEC;
    std::cout << "INFO: _init_arrays() took " <<  time_passed << "s";
    if (tot_memory_MB > 0)
        std::cout << " and used " << tot_memory_MB << "MB of device memory.";
    std::cout << std::endl;
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_synapses_sources;
    f_static_array__array_synapses_sources.open("static_arrays/_static_array__array_synapses_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_sources.is_open())
    {
        f_static_array__array_synapses_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_sources), 27*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_sources, _static_array__array_synapses_sources, sizeof(int32_t)*27, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_sources_1;
    f_static_array__array_synapses_sources_1.open("static_arrays/_static_array__array_synapses_sources_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_sources_1.is_open())
    {
        f_static_array__array_synapses_sources_1.read(reinterpret_cast<char*>(_static_array__array_synapses_sources_1), 243*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_sources_1, _static_array__array_synapses_sources_1, sizeof(int32_t)*243, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_targets;
    f_static_array__array_synapses_targets.open("static_arrays/_static_array__array_synapses_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_targets.is_open())
    {
        f_static_array__array_synapses_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_targets), 27*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_targets, _static_array__array_synapses_targets, sizeof(int32_t)*27, cudaMemcpyHostToDevice)
            );
    ifstream f_static_array__array_synapses_targets_1;
    f_static_array__array_synapses_targets_1.open("static_arrays/_static_array__array_synapses_targets_1", ios::in | ios::binary);
    if(f_static_array__array_synapses_targets_1.is_open())
    {
        f_static_array__array_synapses_targets_1.read(reinterpret_cast<char*>(_static_array__array_synapses_targets_1), 243*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_targets_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_static_array__array_synapses_targets_1, _static_array__array_synapses_targets_1, sizeof(int32_t)*243, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_2_values;
    f_timedarray_2_values.open("static_arrays/_timedarray_2_values", ios::in | ios::binary);
    if(f_timedarray_2_values.is_open())
    {
        f_timedarray_2_values.read(reinterpret_cast<char*>(_timedarray_2_values), 4800*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_2_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_2_values, _timedarray_2_values, sizeof(double)*4800, cudaMemcpyHostToDevice)
            );
    ifstream f_timedarray_values;
    f_timedarray_values.open("static_arrays/_timedarray_values", ios::in | ios::binary);
    if(f_timedarray_values.is_open())
    {
        f_timedarray_values.read(reinterpret_cast<char*>(_timedarray_values), 9000*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _timedarray_values." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(dev_timedarray_values, _timedarray_values, sizeof(double)*9000, cudaMemcpyHostToDevice)
            );
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(double));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(double));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(int64_t));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_A, dev_array_neurongroup_10_A, sizeof(double)*_num__array_neurongroup_10_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_A;
    outfile__array_neurongroup_10_A.open(results_dir + "_array_neurongroup_10_A_2728314578", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_A.is_open())
    {
        outfile__array_neurongroup_10_A.write(reinterpret_cast<char*>(_array_neurongroup_10_A), 16*sizeof(double));
        outfile__array_neurongroup_10_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_batch_sum_X, dev_array_neurongroup_10_batch_sum_X, sizeof(double)*_num__array_neurongroup_10_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_batch_sum_X;
    outfile__array_neurongroup_10_batch_sum_X.open(results_dir + "_array_neurongroup_10_batch_sum_X_4101187682", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_10_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_10_batch_sum_X), 16*sizeof(double));
        outfile__array_neurongroup_10_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_i, dev_array_neurongroup_10_i, sizeof(int32_t)*_num__array_neurongroup_10_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_i;
    outfile__array_neurongroup_10_i.open(results_dir + "_array_neurongroup_10_i_2536205864", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_i.is_open())
    {
        outfile__array_neurongroup_10_i.write(reinterpret_cast<char*>(_array_neurongroup_10_i), 16*sizeof(int32_t));
        outfile__array_neurongroup_10_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_Iexc, dev_array_neurongroup_10_Iexc, sizeof(double)*_num__array_neurongroup_10_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_Iexc;
    outfile__array_neurongroup_10_Iexc.open(results_dir + "_array_neurongroup_10_Iexc_1135481949", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_Iexc.is_open())
    {
        outfile__array_neurongroup_10_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_10_Iexc), 16*sizeof(double));
        outfile__array_neurongroup_10_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_Iinh, dev_array_neurongroup_10_Iinh, sizeof(double)*_num__array_neurongroup_10_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_Iinh;
    outfile__array_neurongroup_10_Iinh.open(results_dir + "_array_neurongroup_10_Iinh_3254681958", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_Iinh.is_open())
    {
        outfile__array_neurongroup_10_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_10_Iinh), 16*sizeof(double));
        outfile__array_neurongroup_10_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_running_sum_X, dev_array_neurongroup_10_running_sum_X, sizeof(double)*_num__array_neurongroup_10_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_running_sum_X;
    outfile__array_neurongroup_10_running_sum_X.open(results_dir + "_array_neurongroup_10_running_sum_X_550445127", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_running_sum_X.is_open())
    {
        outfile__array_neurongroup_10_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_10_running_sum_X), 16*sizeof(double));
        outfile__array_neurongroup_10_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_v, dev_array_neurongroup_10_v, sizeof(double)*_num__array_neurongroup_10_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_v;
    outfile__array_neurongroup_10_v.open(results_dir + "_array_neurongroup_10_v_438526941", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_v.is_open())
    {
        outfile__array_neurongroup_10_v.write(reinterpret_cast<char*>(_array_neurongroup_10_v), 16*sizeof(double));
        outfile__array_neurongroup_10_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_X, dev_array_neurongroup_10_X, sizeof(double)*_num__array_neurongroup_10_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_X;
    outfile__array_neurongroup_10_X.open(results_dir + "_array_neurongroup_10_X_3337973266", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_X.is_open())
    {
        outfile__array_neurongroup_10_X.write(reinterpret_cast<char*>(_array_neurongroup_10_X), 16*sizeof(double));
        outfile__array_neurongroup_10_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_10_Y, dev_array_neurongroup_10_Y, sizeof(double)*_num__array_neurongroup_10_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_10_Y;
    outfile__array_neurongroup_10_Y.open(results_dir + "_array_neurongroup_10_Y_2985451140", ios::binary | ios::out);
    if(outfile__array_neurongroup_10_Y.is_open())
    {
        outfile__array_neurongroup_10_Y.write(reinterpret_cast<char*>(_array_neurongroup_10_Y), 16*sizeof(double));
        outfile__array_neurongroup_10_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_10_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_A, dev_array_neurongroup_11_A, sizeof(double)*_num__array_neurongroup_11_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_A;
    outfile__array_neurongroup_11_A.open(results_dir + "_array_neurongroup_11_A_2740759781", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_A.is_open())
    {
        outfile__array_neurongroup_11_A.write(reinterpret_cast<char*>(_array_neurongroup_11_A), 120*sizeof(double));
        outfile__array_neurongroup_11_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_i, dev_array_neurongroup_11_i, sizeof(int32_t)*_num__array_neurongroup_11_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_i;
    outfile__array_neurongroup_11_i.open(results_dir + "_array_neurongroup_11_i_2531853343", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_i.is_open())
    {
        outfile__array_neurongroup_11_i.write(reinterpret_cast<char*>(_array_neurongroup_11_i), 120*sizeof(int32_t));
        outfile__array_neurongroup_11_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_Iexc, dev_array_neurongroup_11_Iexc, sizeof(double)*_num__array_neurongroup_11_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_Iexc;
    outfile__array_neurongroup_11_Iexc.open(results_dir + "_array_neurongroup_11_Iexc_2297612280", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_Iexc.is_open())
    {
        outfile__array_neurongroup_11_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_11_Iexc), 120*sizeof(double));
        outfile__array_neurongroup_11_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_Iexc2, dev_array_neurongroup_11_Iexc2, sizeof(double)*_num__array_neurongroup_11_Iexc2, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_Iexc2;
    outfile__array_neurongroup_11_Iexc2.open(results_dir + "_array_neurongroup_11_Iexc2_2839230180", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_Iexc2.is_open())
    {
        outfile__array_neurongroup_11_Iexc2.write(reinterpret_cast<char*>(_array_neurongroup_11_Iexc2), 120*sizeof(double));
        outfile__array_neurongroup_11_Iexc2.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_Iexc2." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_indices, dev_array_neurongroup_11_indices, sizeof(int32_t)*_num__array_neurongroup_11_indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_indices;
    outfile__array_neurongroup_11_indices.open(results_dir + "_array_neurongroup_11_indices_1734258926", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_indices.is_open())
    {
        outfile__array_neurongroup_11_indices.write(reinterpret_cast<char*>(_array_neurongroup_11_indices), 120*sizeof(int32_t));
        outfile__array_neurongroup_11_indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_v, dev_array_neurongroup_11_v, sizeof(double)*_num__array_neurongroup_11_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_v;
    outfile__array_neurongroup_11_v.open(results_dir + "_array_neurongroup_11_v_467732970", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_v.is_open())
    {
        outfile__array_neurongroup_11_v.write(reinterpret_cast<char*>(_array_neurongroup_11_v), 120*sizeof(double));
        outfile__array_neurongroup_11_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_X, dev_array_neurongroup_11_X, sizeof(double)*_num__array_neurongroup_11_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_X;
    outfile__array_neurongroup_11_X.open(results_dir + "_array_neurongroup_11_X_3342271525", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_X.is_open())
    {
        outfile__array_neurongroup_11_X.write(reinterpret_cast<char*>(_array_neurongroup_11_X), 120*sizeof(double));
        outfile__array_neurongroup_11_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_11_Y, dev_array_neurongroup_11_Y, sizeof(double)*_num__array_neurongroup_11_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_11_Y;
    outfile__array_neurongroup_11_Y.open(results_dir + "_array_neurongroup_11_Y_2955949235", ios::binary | ios::out);
    if(outfile__array_neurongroup_11_Y.is_open())
    {
        outfile__array_neurongroup_11_Y.write(reinterpret_cast<char*>(_array_neurongroup_11_Y), 120*sizeof(double));
        outfile__array_neurongroup_11_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_11_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_A, dev_array_neurongroup_1_A, sizeof(double)*_num__array_neurongroup_1_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_A;
    outfile__array_neurongroup_1_A.open(results_dir + "_array_neurongroup_1_A_4005009999", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_A.is_open())
    {
        outfile__array_neurongroup_1_A.write(reinterpret_cast<char*>(_array_neurongroup_1_A), 2*sizeof(double));
        outfile__array_neurongroup_1_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_batch_sum_X, dev_array_neurongroup_1_batch_sum_X, sizeof(double)*_num__array_neurongroup_1_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_batch_sum_X;
    outfile__array_neurongroup_1_batch_sum_X.open(results_dir + "_array_neurongroup_1_batch_sum_X_835911465", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_1_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_1_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_1_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_i, dev_array_neurongroup_1_i, sizeof(int32_t)*_num__array_neurongroup_1_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open(results_dir + "_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_Iexc, dev_array_neurongroup_1_Iexc, sizeof(double)*_num__array_neurongroup_1_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_Iexc;
    outfile__array_neurongroup_1_Iexc.open(results_dir + "_array_neurongroup_1_Iexc_3805918598", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_Iexc.is_open())
    {
        outfile__array_neurongroup_1_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_1_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_1_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_Iinh, dev_array_neurongroup_1_Iinh, sizeof(double)*_num__array_neurongroup_1_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_Iinh;
    outfile__array_neurongroup_1_Iinh.open(results_dir + "_array_neurongroup_1_Iinh_1619599549", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_Iinh.is_open())
    {
        outfile__array_neurongroup_1_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_1_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_1_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_running_sum_X, dev_array_neurongroup_1_running_sum_X, sizeof(double)*_num__array_neurongroup_1_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_running_sum_X;
    outfile__array_neurongroup_1_running_sum_X.open(results_dir + "_array_neurongroup_1_running_sum_X_737490015", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_running_sum_X.is_open())
    {
        outfile__array_neurongroup_1_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_1_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_1_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_v, dev_array_neurongroup_1_v, sizeof(double)*_num__array_neurongroup_1_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_v;
    outfile__array_neurongroup_1_v.open(results_dir + "_array_neurongroup_1_v_1443512128", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_v.is_open())
    {
        outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 2*sizeof(double));
        outfile__array_neurongroup_1_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_X, dev_array_neurongroup_1_X, sizeof(double)*_num__array_neurongroup_1_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_X;
    outfile__array_neurongroup_1_X.open(results_dir + "_array_neurongroup_1_X_2329686671", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_X.is_open())
    {
        outfile__array_neurongroup_1_X.write(reinterpret_cast<char*>(_array_neurongroup_1_X), 2*sizeof(double));
        outfile__array_neurongroup_1_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_1_Y, dev_array_neurongroup_1_Y, sizeof(double)*_num__array_neurongroup_1_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_1_Y;
    outfile__array_neurongroup_1_Y.open(results_dir + "_array_neurongroup_1_Y_4258988569", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_Y.is_open())
    {
        outfile__array_neurongroup_1_Y.write(reinterpret_cast<char*>(_array_neurongroup_1_Y), 2*sizeof(double));
        outfile__array_neurongroup_1_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_A, dev_array_neurongroup_2_A, sizeof(double)*_num__array_neurongroup_2_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_A;
    outfile__array_neurongroup_2_A.open(results_dir + "_array_neurongroup_2_A_3975226390", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_A.is_open())
    {
        outfile__array_neurongroup_2_A.write(reinterpret_cast<char*>(_array_neurongroup_2_A), 2*sizeof(double));
        outfile__array_neurongroup_2_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_batch_sum_X, dev_array_neurongroup_2_batch_sum_X, sizeof(double)*_num__array_neurongroup_2_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_batch_sum_X;
    outfile__array_neurongroup_2_batch_sum_X.open(results_dir + "_array_neurongroup_2_batch_sum_X_2350420967", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_2_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_2_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_2_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_i, dev_array_neurongroup_2_i, sizeof(int32_t)*_num__array_neurongroup_2_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_i;
    outfile__array_neurongroup_2_i.open(results_dir + "_array_neurongroup_2_i_3645148396", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_i.is_open())
    {
        outfile__array_neurongroup_2_i.write(reinterpret_cast<char*>(_array_neurongroup_2_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_2_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_Iexc, dev_array_neurongroup_2_Iexc, sizeof(double)*_num__array_neurongroup_2_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_Iexc;
    outfile__array_neurongroup_2_Iexc.open(results_dir + "_array_neurongroup_2_Iexc_1682818856", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_Iexc.is_open())
    {
        outfile__array_neurongroup_2_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_2_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_2_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_Iinh, dev_array_neurongroup_2_Iinh, sizeof(double)*_num__array_neurongroup_2_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_Iinh;
    outfile__array_neurongroup_2_Iinh.open(results_dir + "_array_neurongroup_2_Iinh_3860681235", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_Iinh.is_open())
    {
        outfile__array_neurongroup_2_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_2_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_2_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_running_sum_X, dev_array_neurongroup_2_running_sum_X, sizeof(double)*_num__array_neurongroup_2_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_running_sum_X;
    outfile__array_neurongroup_2_running_sum_X.open(results_dir + "_array_neurongroup_2_running_sum_X_982015526", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_running_sum_X.is_open())
    {
        outfile__array_neurongroup_2_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_2_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_2_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_v, dev_array_neurongroup_2_v, sizeof(double)*_num__array_neurongroup_2_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_v;
    outfile__array_neurongroup_2_v.open(results_dir + "_array_neurongroup_2_v_1414299929", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_v.is_open())
    {
        outfile__array_neurongroup_2_v.write(reinterpret_cast<char*>(_array_neurongroup_2_v), 2*sizeof(double));
        outfile__array_neurongroup_2_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_X, dev_array_neurongroup_2_X, sizeof(double)*_num__array_neurongroup_2_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_X;
    outfile__array_neurongroup_2_X.open(results_dir + "_array_neurongroup_2_X_2291829974", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_X.is_open())
    {
        outfile__array_neurongroup_2_X.write(reinterpret_cast<char*>(_array_neurongroup_2_X), 2*sizeof(double));
        outfile__array_neurongroup_2_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_2_Y, dev_array_neurongroup_2_Y, sizeof(double)*_num__array_neurongroup_2_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_2_Y;
    outfile__array_neurongroup_2_Y.open(results_dir + "_array_neurongroup_2_Y_4288527424", ios::binary | ios::out);
    if(outfile__array_neurongroup_2_Y.is_open())
    {
        outfile__array_neurongroup_2_Y.write(reinterpret_cast<char*>(_array_neurongroup_2_Y), 2*sizeof(double));
        outfile__array_neurongroup_2_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_2_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_A, dev_array_neurongroup_3_A, sizeof(double)*_num__array_neurongroup_3_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_A;
    outfile__array_neurongroup_3_A.open(results_dir + "_array_neurongroup_3_A_3979562529", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_A.is_open())
    {
        outfile__array_neurongroup_3_A.write(reinterpret_cast<char*>(_array_neurongroup_3_A), 2*sizeof(double));
        outfile__array_neurongroup_3_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_batch_sum_X, dev_array_neurongroup_3_batch_sum_X, sizeof(double)*_num__array_neurongroup_3_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_batch_sum_X;
    outfile__array_neurongroup_3_batch_sum_X.open(results_dir + "_array_neurongroup_3_batch_sum_X_1368279650", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_3_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_3_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_3_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_i, dev_array_neurongroup_3_i, sizeof(int32_t)*_num__array_neurongroup_3_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_i;
    outfile__array_neurongroup_3_i.open(results_dir + "_array_neurongroup_3_i_3632719579", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_i.is_open())
    {
        outfile__array_neurongroup_3_i.write(reinterpret_cast<char*>(_array_neurongroup_3_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_3_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_Iexc, dev_array_neurongroup_3_Iexc, sizeof(double)*_num__array_neurongroup_3_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_Iexc;
    outfile__array_neurongroup_3_Iexc.open(results_dir + "_array_neurongroup_3_Iexc_2937132173", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_Iexc.is_open())
    {
        outfile__array_neurongroup_3_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_3_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_3_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_Iinh, dev_array_neurongroup_3_Iinh, sizeof(double)*_num__array_neurongroup_3_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_Iinh;
    outfile__array_neurongroup_3_Iinh.open(results_dir + "_array_neurongroup_3_Iinh_759267766", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_Iinh.is_open())
    {
        outfile__array_neurongroup_3_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_3_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_3_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_running_sum_X, dev_array_neurongroup_3_running_sum_X, sizeof(double)*_num__array_neurongroup_3_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_running_sum_X;
    outfile__array_neurongroup_3_running_sum_X.open(results_dir + "_array_neurongroup_3_running_sum_X_2205386190", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_running_sum_X.is_open())
    {
        outfile__array_neurongroup_3_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_3_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_3_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_v, dev_array_neurongroup_3_v, sizeof(double)*_num__array_neurongroup_3_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_v;
    outfile__array_neurongroup_3_v.open(results_dir + "_array_neurongroup_3_v_1435429678", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_v.is_open())
    {
        outfile__array_neurongroup_3_v.write(reinterpret_cast<char*>(_array_neurongroup_3_v), 2*sizeof(double));
        outfile__array_neurongroup_3_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_X, dev_array_neurongroup_3_X, sizeof(double)*_num__array_neurongroup_3_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_X;
    outfile__array_neurongroup_3_X.open(results_dir + "_array_neurongroup_3_X_2304304865", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_X.is_open())
    {
        outfile__array_neurongroup_3_X.write(reinterpret_cast<char*>(_array_neurongroup_3_X), 2*sizeof(double));
        outfile__array_neurongroup_3_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_3_Y, dev_array_neurongroup_3_Y, sizeof(double)*_num__array_neurongroup_3_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_3_Y;
    outfile__array_neurongroup_3_Y.open(results_dir + "_array_neurongroup_3_Y_4267693687", ios::binary | ios::out);
    if(outfile__array_neurongroup_3_Y.is_open())
    {
        outfile__array_neurongroup_3_Y.write(reinterpret_cast<char*>(_array_neurongroup_3_Y), 2*sizeof(double));
        outfile__array_neurongroup_3_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_3_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_A, dev_array_neurongroup_4_A, sizeof(double)*_num__array_neurongroup_4_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_A;
    outfile__array_neurongroup_4_A.open(results_dir + "_array_neurongroup_4_A_3900463268", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_A.is_open())
    {
        outfile__array_neurongroup_4_A.write(reinterpret_cast<char*>(_array_neurongroup_4_A), 2*sizeof(double));
        outfile__array_neurongroup_4_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_batch_sum_X, dev_array_neurongroup_4_batch_sum_X, sizeof(double)*_num__array_neurongroup_4_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_batch_sum_X;
    outfile__array_neurongroup_4_batch_sum_X.open(results_dir + "_array_neurongroup_4_batch_sum_X_754797626", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_4_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_4_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_4_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_i, dev_array_neurongroup_4_i, sizeof(int32_t)*_num__array_neurongroup_4_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_i;
    outfile__array_neurongroup_4_i.open(results_dir + "_array_neurongroup_4_i_3720999006", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_i.is_open())
    {
        outfile__array_neurongroup_4_i.write(reinterpret_cast<char*>(_array_neurongroup_4_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_4_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_Iexc, dev_array_neurongroup_4_Iexc, sizeof(double)*_num__array_neurongroup_4_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_Iexc;
    outfile__array_neurongroup_4_Iexc.open(results_dir + "_array_neurongroup_4_Iexc_2987664437", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_Iexc.is_open())
    {
        outfile__array_neurongroup_4_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_4_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_4_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_Iinh, dev_array_neurongroup_4_Iinh, sizeof(double)*_num__array_neurongroup_4_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_Iinh;
    outfile__array_neurongroup_4_Iinh.open(results_dir + "_array_neurongroup_4_Iinh_809808142", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_Iinh.is_open())
    {
        outfile__array_neurongroup_4_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_4_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_4_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_running_sum_X, dev_array_neurongroup_4_running_sum_X, sizeof(double)*_num__array_neurongroup_4_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_running_sum_X;
    outfile__array_neurongroup_4_running_sum_X.open(results_dir + "_array_neurongroup_4_running_sum_X_410159828", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_running_sum_X.is_open())
    {
        outfile__array_neurongroup_4_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_4_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_4_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_v, dev_array_neurongroup_4_v, sizeof(double)*_num__array_neurongroup_4_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_v;
    outfile__array_neurongroup_4_v.open(results_dir + "_array_neurongroup_4_v_1354890667", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_v.is_open())
    {
        outfile__array_neurongroup_4_v.write(reinterpret_cast<char*>(_array_neurongroup_4_v), 2*sizeof(double));
        outfile__array_neurongroup_4_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_X, dev_array_neurongroup_4_X, sizeof(double)*_num__array_neurongroup_4_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_X;
    outfile__array_neurongroup_4_X.open(results_dir + "_array_neurongroup_4_X_2350379108", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_X.is_open())
    {
        outfile__array_neurongroup_4_X.write(reinterpret_cast<char*>(_array_neurongroup_4_X), 2*sizeof(double));
        outfile__array_neurongroup_4_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_4_Y, dev_array_neurongroup_4_Y, sizeof(double)*_num__array_neurongroup_4_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_4_Y;
    outfile__array_neurongroup_4_Y.open(results_dir + "_array_neurongroup_4_Y_4212179186", ios::binary | ios::out);
    if(outfile__array_neurongroup_4_Y.is_open())
    {
        outfile__array_neurongroup_4_Y.write(reinterpret_cast<char*>(_array_neurongroup_4_Y), 2*sizeof(double));
        outfile__array_neurongroup_4_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_4_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_A, dev_array_neurongroup_5_A, sizeof(double)*_num__array_neurongroup_5_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_A;
    outfile__array_neurongroup_5_A.open(results_dir + "_array_neurongroup_5_A_3921556115", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_A.is_open())
    {
        outfile__array_neurongroup_5_A.write(reinterpret_cast<char*>(_array_neurongroup_5_A), 2*sizeof(double));
        outfile__array_neurongroup_5_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_batch_sum_X, dev_array_neurongroup_5_batch_sum_X, sizeof(double)*_num__array_neurongroup_5_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_batch_sum_X;
    outfile__array_neurongroup_5_batch_sum_X.open(results_dir + "_array_neurongroup_5_batch_sum_X_4050359743", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_5_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_5_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_5_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_i, dev_array_neurongroup_5_i, sizeof(int32_t)*_num__array_neurongroup_5_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_i;
    outfile__array_neurongroup_5_i.open(results_dir + "_array_neurongroup_5_i_3691747945", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_i.is_open())
    {
        outfile__array_neurongroup_5_i.write(reinterpret_cast<char*>(_array_neurongroup_5_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_5_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_Iexc, dev_array_neurongroup_5_Iexc, sizeof(double)*_num__array_neurongroup_5_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_Iexc;
    outfile__array_neurongroup_5_Iexc.open(results_dir + "_array_neurongroup_5_Iexc_2034825104", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_Iexc.is_open())
    {
        outfile__array_neurongroup_5_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_5_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_5_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_Iinh, dev_array_neurongroup_5_Iinh, sizeof(double)*_num__array_neurongroup_5_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_Iinh;
    outfile__array_neurongroup_5_Iinh.open(results_dir + "_array_neurongroup_5_Iinh_4212679339", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_Iinh.is_open())
    {
        outfile__array_neurongroup_5_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_5_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_5_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_running_sum_X, dev_array_neurongroup_5_running_sum_X, sizeof(double)*_num__array_neurongroup_5_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_running_sum_X;
    outfile__array_neurongroup_5_running_sum_X.open(results_dir + "_array_neurongroup_5_running_sum_X_2710131004", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_running_sum_X.is_open())
    {
        outfile__array_neurongroup_5_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_5_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_5_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_v, dev_array_neurongroup_5_v, sizeof(double)*_num__array_neurongroup_5_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_v;
    outfile__array_neurongroup_5_v.open(results_dir + "_array_neurongroup_5_v_1359189916", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_v.is_open())
    {
        outfile__array_neurongroup_5_v.write(reinterpret_cast<char*>(_array_neurongroup_5_v), 2*sizeof(double));
        outfile__array_neurongroup_5_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_X, dev_array_neurongroup_5_X, sizeof(double)*_num__array_neurongroup_5_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_X;
    outfile__array_neurongroup_5_X.open(results_dir + "_array_neurongroup_5_X_2379586131", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_X.is_open())
    {
        outfile__array_neurongroup_5_X.write(reinterpret_cast<char*>(_array_neurongroup_5_X), 2*sizeof(double));
        outfile__array_neurongroup_5_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_5_Y, dev_array_neurongroup_5_Y, sizeof(double)*_num__array_neurongroup_5_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_5_Y;
    outfile__array_neurongroup_5_Y.open(results_dir + "_array_neurongroup_5_Y_4208110277", ios::binary | ios::out);
    if(outfile__array_neurongroup_5_Y.is_open())
    {
        outfile__array_neurongroup_5_Y.write(reinterpret_cast<char*>(_array_neurongroup_5_Y), 2*sizeof(double));
        outfile__array_neurongroup_5_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_5_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_A, dev_array_neurongroup_6_A, sizeof(double)*_num__array_neurongroup_6_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_A;
    outfile__array_neurongroup_6_A.open(results_dir + "_array_neurongroup_6_A_3958934730", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_A.is_open())
    {
        outfile__array_neurongroup_6_A.write(reinterpret_cast<char*>(_array_neurongroup_6_A), 2*sizeof(double));
        outfile__array_neurongroup_6_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_batch_sum_X, dev_array_neurongroup_6_batch_sum_X, sizeof(double)*_num__array_neurongroup_6_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_batch_sum_X;
    outfile__array_neurongroup_6_batch_sum_X.open(results_dir + "_array_neurongroup_6_batch_sum_X_1285683569", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_6_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_6_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_6_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_i, dev_array_neurongroup_6_i, sizeof(int32_t)*_num__array_neurongroup_6_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_i;
    outfile__array_neurongroup_6_i.open(results_dir + "_array_neurongroup_6_i_3729597488", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_i.is_open())
    {
        outfile__array_neurongroup_6_i.write(reinterpret_cast<char*>(_array_neurongroup_6_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_6_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_Iexc, dev_array_neurongroup_6_Iexc, sizeof(double)*_num__array_neurongroup_6_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_Iexc;
    outfile__array_neurongroup_6_Iexc.open(results_dir + "_array_neurongroup_6_Iexc_4292642110", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_Iexc.is_open())
    {
        outfile__array_neurongroup_6_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_6_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_6_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_Iinh, dev_array_neurongroup_6_Iinh, sizeof(double)*_num__array_neurongroup_6_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_Iinh;
    outfile__array_neurongroup_6_Iinh.open(results_dir + "_array_neurongroup_6_Iinh_2106331141", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_Iinh.is_open())
    {
        outfile__array_neurongroup_6_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_6_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_6_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_running_sum_X, dev_array_neurongroup_6_running_sum_X, sizeof(double)*_num__array_neurongroup_6_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_running_sum_X;
    outfile__array_neurongroup_6_running_sum_X.open(results_dir + "_array_neurongroup_6_running_sum_X_2968795973", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_running_sum_X.is_open())
    {
        outfile__array_neurongroup_6_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_6_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_6_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_v, dev_array_neurongroup_6_v, sizeof(double)*_num__array_neurongroup_6_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_v;
    outfile__array_neurongroup_6_v.open(results_dir + "_array_neurongroup_6_v_1397041605", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_v.is_open())
    {
        outfile__array_neurongroup_6_v.write(reinterpret_cast<char*>(_array_neurongroup_6_v), 2*sizeof(double));
        outfile__array_neurongroup_6_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_X, dev_array_neurongroup_6_X, sizeof(double)*_num__array_neurongroup_6_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_X;
    outfile__array_neurongroup_6_X.open(results_dir + "_array_neurongroup_6_X_2408784906", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_X.is_open())
    {
        outfile__array_neurongroup_6_X.write(reinterpret_cast<char*>(_array_neurongroup_6_X), 2*sizeof(double));
        outfile__array_neurongroup_6_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_6_Y, dev_array_neurongroup_6_Y, sizeof(double)*_num__array_neurongroup_6_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_6_Y;
    outfile__array_neurongroup_6_Y.open(results_dir + "_array_neurongroup_6_Y_4170454172", ios::binary | ios::out);
    if(outfile__array_neurongroup_6_Y.is_open())
    {
        outfile__array_neurongroup_6_Y.write(reinterpret_cast<char*>(_array_neurongroup_6_Y), 2*sizeof(double));
        outfile__array_neurongroup_6_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_6_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_A, dev_array_neurongroup_7_A, sizeof(double)*_num__array_neurongroup_7_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_A;
    outfile__array_neurongroup_7_A.open(results_dir + "_array_neurongroup_7_A_3929728765", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_A.is_open())
    {
        outfile__array_neurongroup_7_A.write(reinterpret_cast<char*>(_array_neurongroup_7_A), 2*sizeof(double));
        outfile__array_neurongroup_7_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_batch_sum_X, dev_array_neurongroup_7_batch_sum_X, sizeof(double)*_num__array_neurongroup_7_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_batch_sum_X;
    outfile__array_neurongroup_7_batch_sum_X.open(results_dir + "_array_neurongroup_7_batch_sum_X_2436309236", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_7_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_7_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_7_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_i, dev_array_neurongroup_7_i, sizeof(int32_t)*_num__array_neurongroup_7_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_i;
    outfile__array_neurongroup_7_i.open(results_dir + "_array_neurongroup_7_i_3750710791", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_i.is_open())
    {
        outfile__array_neurongroup_7_i.write(reinterpret_cast<char*>(_array_neurongroup_7_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_7_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_Iexc, dev_array_neurongroup_7_Iexc, sizeof(double)*_num__array_neurongroup_7_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_Iexc;
    outfile__array_neurongroup_7_Iexc.open(results_dir + "_array_neurongroup_7_Iexc_880826011", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_Iexc.is_open())
    {
        outfile__array_neurongroup_7_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_7_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_7_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_Iinh, dev_array_neurongroup_7_Iinh, sizeof(double)*_num__array_neurongroup_7_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_Iinh;
    outfile__array_neurongroup_7_Iinh.open(results_dir + "_array_neurongroup_7_Iinh_3067134880", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_Iinh.is_open())
    {
        outfile__array_neurongroup_7_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_7_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_7_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_running_sum_X, dev_array_neurongroup_7_running_sum_X, sizeof(double)*_num__array_neurongroup_7_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_running_sum_X;
    outfile__array_neurongroup_7_running_sum_X.open(results_dir + "_array_neurongroup_7_running_sum_X_152035501", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_running_sum_X.is_open())
    {
        outfile__array_neurongroup_7_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_7_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_7_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_v, dev_array_neurongroup_7_v, sizeof(double)*_num__array_neurongroup_7_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_v;
    outfile__array_neurongroup_7_v.open(results_dir + "_array_neurongroup_7_v_1384596466", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_v.is_open())
    {
        outfile__array_neurongroup_7_v.write(reinterpret_cast<char*>(_array_neurongroup_7_v), 2*sizeof(double));
        outfile__array_neurongroup_7_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_X, dev_array_neurongroup_7_X, sizeof(double)*_num__array_neurongroup_7_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_X;
    outfile__array_neurongroup_7_X.open(results_dir + "_array_neurongroup_7_X_2387693117", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_X.is_open())
    {
        outfile__array_neurongroup_7_X.write(reinterpret_cast<char*>(_array_neurongroup_7_X), 2*sizeof(double));
        outfile__array_neurongroup_7_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_7_Y, dev_array_neurongroup_7_Y, sizeof(double)*_num__array_neurongroup_7_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_7_Y;
    outfile__array_neurongroup_7_Y.open(results_dir + "_array_neurongroup_7_Y_4183195307", ios::binary | ios::out);
    if(outfile__array_neurongroup_7_Y.is_open())
    {
        outfile__array_neurongroup_7_Y.write(reinterpret_cast<char*>(_array_neurongroup_7_Y), 2*sizeof(double));
        outfile__array_neurongroup_7_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_7_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_A, dev_array_neurongroup_8_A, sizeof(double)*_num__array_neurongroup_8_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_A;
    outfile__array_neurongroup_8_A.open(results_dir + "_array_neurongroup_8_A_3781599680", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_A.is_open())
    {
        outfile__array_neurongroup_8_A.write(reinterpret_cast<char*>(_array_neurongroup_8_A), 2*sizeof(double));
        outfile__array_neurongroup_8_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_batch_sum_X, dev_array_neurongroup_8_batch_sum_X, sizeof(double)*_num__array_neurongroup_8_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_batch_sum_X;
    outfile__array_neurongroup_8_batch_sum_X.open(results_dir + "_array_neurongroup_8_batch_sum_X_3058169281", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_8_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_8_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_8_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_i, dev_array_neurongroup_8_i, sizeof(int32_t)*_num__array_neurongroup_8_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_i;
    outfile__array_neurongroup_8_i.open(results_dir + "_array_neurongroup_8_i_3570600250", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_i.is_open())
    {
        outfile__array_neurongroup_8_i.write(reinterpret_cast<char*>(_array_neurongroup_8_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_8_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_Iexc, dev_array_neurongroup_8_Iexc, sizeof(double)*_num__array_neurongroup_8_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_Iexc;
    outfile__array_neurongroup_8_Iexc.open(results_dir + "_array_neurongroup_8_Iexc_3319194702", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_Iexc.is_open())
    {
        outfile__array_neurongroup_8_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_8_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_8_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_Iinh, dev_array_neurongroup_8_Iinh, sizeof(double)*_num__array_neurongroup_8_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_Iinh;
    outfile__array_neurongroup_8_Iinh.open(results_dir + "_array_neurongroup_8_Iinh_1199993205", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_Iinh.is_open())
    {
        outfile__array_neurongroup_8_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_8_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_8_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_running_sum_X, dev_array_neurongroup_8_running_sum_X, sizeof(double)*_num__array_neurongroup_8_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_running_sum_X;
    outfile__array_neurongroup_8_running_sum_X.open(results_dir + "_array_neurongroup_8_running_sum_X_1569137456", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_running_sum_X.is_open())
    {
        outfile__array_neurongroup_8_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_8_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_8_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_v, dev_array_neurongroup_8_v, sizeof(double)*_num__array_neurongroup_8_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_v;
    outfile__array_neurongroup_8_v.open(results_dir + "_array_neurongroup_8_v_1507525839", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_v.is_open())
    {
        outfile__array_neurongroup_8_v.write(reinterpret_cast<char*>(_array_neurongroup_8_v), 2*sizeof(double));
        outfile__array_neurongroup_8_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_X, dev_array_neurongroup_8_X, sizeof(double)*_num__array_neurongroup_8_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_X;
    outfile__array_neurongroup_8_X.open(results_dir + "_array_neurongroup_8_X_2232224000", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_X.is_open())
    {
        outfile__array_neurongroup_8_X.write(reinterpret_cast<char*>(_array_neurongroup_8_X), 2*sizeof(double));
        outfile__array_neurongroup_8_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_8_Y, dev_array_neurongroup_8_Y, sizeof(double)*_num__array_neurongroup_8_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_8_Y;
    outfile__array_neurongroup_8_Y.open(results_dir + "_array_neurongroup_8_Y_4060756374", ios::binary | ios::out);
    if(outfile__array_neurongroup_8_Y.is_open())
    {
        outfile__array_neurongroup_8_Y.write(reinterpret_cast<char*>(_array_neurongroup_8_Y), 2*sizeof(double));
        outfile__array_neurongroup_8_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_8_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_A, dev_array_neurongroup_9_A, sizeof(double)*_num__array_neurongroup_9_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_A;
    outfile__array_neurongroup_9_A.open(results_dir + "_array_neurongroup_9_A_3768896503", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_A.is_open())
    {
        outfile__array_neurongroup_9_A.write(reinterpret_cast<char*>(_array_neurongroup_9_A), 2*sizeof(double));
        outfile__array_neurongroup_9_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_batch_sum_X, dev_array_neurongroup_9_batch_sum_X, sizeof(double)*_num__array_neurongroup_9_batch_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_batch_sum_X;
    outfile__array_neurongroup_9_batch_sum_X.open(results_dir + "_array_neurongroup_9_batch_sum_X_1808870468", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_batch_sum_X.is_open())
    {
        outfile__array_neurongroup_9_batch_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_9_batch_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_9_batch_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_batch_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_i, dev_array_neurongroup_9_i, sizeof(int32_t)*_num__array_neurongroup_9_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_i;
    outfile__array_neurongroup_9_i.open(results_dir + "_array_neurongroup_9_i_3574686477", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_i.is_open())
    {
        outfile__array_neurongroup_9_i.write(reinterpret_cast<char*>(_array_neurongroup_9_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_9_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_Iexc, dev_array_neurongroup_9_Iexc, sizeof(double)*_num__array_neurongroup_9_Iexc, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_Iexc;
    outfile__array_neurongroup_9_Iexc.open(results_dir + "_array_neurongroup_9_Iexc_243939307", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_Iexc.is_open())
    {
        outfile__array_neurongroup_9_Iexc.write(reinterpret_cast<char*>(_array_neurongroup_9_Iexc), 2*sizeof(double));
        outfile__array_neurongroup_9_Iexc.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_Iexc." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_Iinh, dev_array_neurongroup_9_Iinh, sizeof(double)*_num__array_neurongroup_9_Iinh, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_Iinh;
    outfile__array_neurongroup_9_Iinh.open(results_dir + "_array_neurongroup_9_Iinh_2363138768", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_Iinh.is_open())
    {
        outfile__array_neurongroup_9_Iinh.write(reinterpret_cast<char*>(_array_neurongroup_9_Iinh), 2*sizeof(double));
        outfile__array_neurongroup_9_Iinh.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_Iinh." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_running_sum_X, dev_array_neurongroup_9_running_sum_X, sizeof(double)*_num__array_neurongroup_9_running_sum_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_running_sum_X;
    outfile__array_neurongroup_9_running_sum_X.open(results_dir + "_array_neurongroup_9_running_sum_X_3833395416", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_running_sum_X.is_open())
    {
        outfile__array_neurongroup_9_running_sum_X.write(reinterpret_cast<char*>(_array_neurongroup_9_running_sum_X), 2*sizeof(double));
        outfile__array_neurongroup_9_running_sum_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_running_sum_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_v, dev_array_neurongroup_9_v, sizeof(double)*_num__array_neurongroup_9_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_v;
    outfile__array_neurongroup_9_v.open(results_dir + "_array_neurongroup_9_v_1478061816", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_v.is_open())
    {
        outfile__array_neurongroup_9_v.write(reinterpret_cast<char*>(_array_neurongroup_9_v), 2*sizeof(double));
        outfile__array_neurongroup_9_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_X, dev_array_neurongroup_9_X, sizeof(double)*_num__array_neurongroup_9_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_X;
    outfile__array_neurongroup_9_X.open(results_dir + "_array_neurongroup_9_X_2228183863", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_X.is_open())
    {
        outfile__array_neurongroup_9_X.write(reinterpret_cast<char*>(_array_neurongroup_9_X), 2*sizeof(double));
        outfile__array_neurongroup_9_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_9_Y, dev_array_neurongroup_9_Y, sizeof(double)*_num__array_neurongroup_9_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_9_Y;
    outfile__array_neurongroup_9_Y.open(results_dir + "_array_neurongroup_9_Y_4089992097", ios::binary | ios::out);
    if(outfile__array_neurongroup_9_Y.is_open())
    {
        outfile__array_neurongroup_9_Y.write(reinterpret_cast<char*>(_array_neurongroup_9_Y), 2*sizeof(double));
        outfile__array_neurongroup_9_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_9_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_A, dev_array_neurongroup_A, sizeof(double)*_num__array_neurongroup_A, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_A;
    outfile__array_neurongroup_A.open(results_dir + "_array_neurongroup_A_2823903610", ios::binary | ios::out);
    if(outfile__array_neurongroup_A.is_open())
    {
        outfile__array_neurongroup_A.write(reinterpret_cast<char*>(_array_neurongroup_A), 2*sizeof(double));
        outfile__array_neurongroup_A.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_A." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open(results_dir + "_array_neurongroup_i_2649026944", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 2*sizeof(int32_t));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_indices, dev_array_neurongroup_indices, sizeof(int32_t)*_num__array_neurongroup_indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_indices;
    outfile__array_neurongroup_indices.open(results_dir + "_array_neurongroup_indices_4119198825", ios::binary | ios::out);
    if(outfile__array_neurongroup_indices.is_open())
    {
        outfile__array_neurongroup_indices.write(reinterpret_cast<char*>(_array_neurongroup_indices), 2*sizeof(int32_t));
        outfile__array_neurongroup_indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_indices." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_v;
    outfile__array_neurongroup_v.open(results_dir + "_array_neurongroup_v_283966581", ios::binary | ios::out);
    if(outfile__array_neurongroup_v.is_open())
    {
        outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 2*sizeof(double));
        outfile__array_neurongroup_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_v." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_X, dev_array_neurongroup_X, sizeof(double)*_num__array_neurongroup_X, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_X;
    outfile__array_neurongroup_X.open(results_dir + "_array_neurongroup_X_3426415034", ios::binary | ios::out);
    if(outfile__array_neurongroup_X.is_open())
    {
        outfile__array_neurongroup_X.write(reinterpret_cast<char*>(_array_neurongroup_X), 2*sizeof(double));
        outfile__array_neurongroup_X.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_X." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_neurongroup_Y, dev_array_neurongroup_Y, sizeof(double)*_num__array_neurongroup_Y, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_neurongroup_Y;
    outfile__array_neurongroup_Y.open(results_dir + "_array_neurongroup_Y_3141386540", ios::binary | ios::out);
    if(outfile__array_neurongroup_Y.is_open())
    {
        outfile__array_neurongroup_Y.write(reinterpret_cast<char*>(_array_neurongroup_Y), 2*sizeof(double));
        outfile__array_neurongroup_Y.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_Y." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor__source_idx, dev_array_spikemonitor__source_idx, sizeof(int32_t)*_num__array_spikemonitor__source_idx, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open(results_dir + "_array_spikemonitor__source_idx_1477951789", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 2*sizeof(int32_t));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_spikemonitor_count, dev_array_spikemonitor_count, sizeof(int32_t)*_num__array_spikemonitor_count, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open(results_dir + "_array_spikemonitor_count_598337445", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 2*sizeof(int32_t));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open(results_dir + "_array_spikemonitor_N_225734567", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(int32_t));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_statemonitor__indices, dev_array_statemonitor__indices, sizeof(int32_t)*_num__array_statemonitor__indices, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_statemonitor__indices;
    outfile__array_statemonitor__indices.open(results_dir + "_array_statemonitor__indices_2854283999", ios::binary | ios::out);
    if(outfile__array_statemonitor__indices.is_open())
    {
        outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 1*sizeof(int32_t));
        outfile__array_statemonitor__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
    }
    ofstream outfile__array_statemonitor_N;
    outfile__array_statemonitor_N.open(results_dir + "_array_statemonitor_N_4140778434", ios::binary | ios::out);
    if(outfile__array_statemonitor_N.is_open())
    {
        outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(int32_t));
        outfile__array_statemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_10_N, dev_array_synapses_10_N, sizeof(int32_t)*_num__array_synapses_10_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_10_N;
    outfile__array_synapses_10_N.open(results_dir + "_array_synapses_10_N_2387659436", ios::binary | ios::out);
    if(outfile__array_synapses_10_N.is_open())
    {
        outfile__array_synapses_10_N.write(reinterpret_cast<char*>(_array_synapses_10_N), 1*sizeof(int32_t));
        outfile__array_synapses_10_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_10_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_11_N, dev_array_synapses_11_N, sizeof(int32_t)*_num__array_synapses_11_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_11_N;
    outfile__array_synapses_11_N.open(results_dir + "_array_synapses_11_N_2408751259", ios::binary | ios::out);
    if(outfile__array_synapses_11_N.is_open())
    {
        outfile__array_synapses_11_N.write(reinterpret_cast<char*>(_array_synapses_11_N), 1*sizeof(int32_t));
        outfile__array_synapses_11_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_11_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_12_N, dev_array_synapses_12_N, sizeof(int32_t)*_num__array_synapses_12_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_12_N;
    outfile__array_synapses_12_N.open(results_dir + "_array_synapses_12_N_2379488962", ios::binary | ios::out);
    if(outfile__array_synapses_12_N.is_open())
    {
        outfile__array_synapses_12_N.write(reinterpret_cast<char*>(_array_synapses_12_N), 1*sizeof(int32_t));
        outfile__array_synapses_12_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_12_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_13_N, dev_array_synapses_13_N, sizeof(int32_t)*_num__array_synapses_13_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_13_N;
    outfile__array_synapses_13_N.open(results_dir + "_array_synapses_13_N_2350281973", ios::binary | ios::out);
    if(outfile__array_synapses_13_N.is_open())
    {
        outfile__array_synapses_13_N.write(reinterpret_cast<char*>(_array_synapses_13_N), 1*sizeof(int32_t));
        outfile__array_synapses_13_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_13_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_14_N, dev_array_synapses_14_N, sizeof(int32_t)*_num__array_synapses_14_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_14_N;
    outfile__array_synapses_14_N.open(results_dir + "_array_synapses_14_N_2304336496", ios::binary | ios::out);
    if(outfile__array_synapses_14_N.is_open())
    {
        outfile__array_synapses_14_N.write(reinterpret_cast<char*>(_array_synapses_14_N), 1*sizeof(int32_t));
        outfile__array_synapses_14_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_14_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_15_N, dev_array_synapses_15_N, sizeof(int32_t)*_num__array_synapses_15_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_15_N;
    outfile__array_synapses_15_N.open(results_dir + "_array_synapses_15_N_2291861575", ios::binary | ios::out);
    if(outfile__array_synapses_15_N.is_open())
    {
        outfile__array_synapses_15_N.write(reinterpret_cast<char*>(_array_synapses_15_N), 1*sizeof(int32_t));
        outfile__array_synapses_15_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_15_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_1_N, dev_array_synapses_1_N, sizeof(int32_t)*_num__array_synapses_1_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open(results_dir + "_array_synapses_1_N_1771729519", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(int32_t));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_2_N, dev_array_synapses_2_N, sizeof(int32_t)*_num__array_synapses_2_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open(results_dir + "_array_synapses_2_N_1809632310", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(int32_t));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_3_N, dev_array_synapses_3_N, sizeof(int32_t)*_num__array_synapses_3_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_3_N;
    outfile__array_synapses_3_N.open(results_dir + "_array_synapses_3_N_1780393473", ios::binary | ios::out);
    if(outfile__array_synapses_3_N.is_open())
    {
        outfile__array_synapses_3_N.write(reinterpret_cast<char*>(_array_synapses_3_N), 1*sizeof(int32_t));
        outfile__array_synapses_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_4_N, dev_array_synapses_4_N, sizeof(int32_t)*_num__array_synapses_4_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_4_N;
    outfile__array_synapses_4_N.open(results_dir + "_array_synapses_4_N_1867624580", ios::binary | ios::out);
    if(outfile__array_synapses_4_N.is_open())
    {
        outfile__array_synapses_4_N.write(reinterpret_cast<char*>(_array_synapses_4_N), 1*sizeof(int32_t));
        outfile__array_synapses_4_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_5_N, dev_array_synapses_5_N, sizeof(int32_t)*_num__array_synapses_5_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_5_N;
    outfile__array_synapses_5_N.open(results_dir + "_array_synapses_5_N_1855183539", ios::binary | ios::out);
    if(outfile__array_synapses_5_N.is_open())
    {
        outfile__array_synapses_5_N.write(reinterpret_cast<char*>(_array_synapses_5_N), 1*sizeof(int32_t));
        outfile__array_synapses_5_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_6_N, dev_array_synapses_6_N, sizeof(int32_t)*_num__array_synapses_6_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_6_N;
    outfile__array_synapses_6_N.open(results_dir + "_array_synapses_6_N_1825924330", ios::binary | ios::out);
    if(outfile__array_synapses_6_N.is_open())
    {
        outfile__array_synapses_6_N.write(reinterpret_cast<char*>(_array_synapses_6_N), 1*sizeof(int32_t));
        outfile__array_synapses_6_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_6_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_7_N, dev_array_synapses_7_N, sizeof(int32_t)*_num__array_synapses_7_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_7_N;
    outfile__array_synapses_7_N.open(results_dir + "_array_synapses_7_N_1830227677", ios::binary | ios::out);
    if(outfile__array_synapses_7_N.is_open())
    {
        outfile__array_synapses_7_N.write(reinterpret_cast<char*>(_array_synapses_7_N), 1*sizeof(int32_t));
        outfile__array_synapses_7_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_7_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_8_N, dev_array_synapses_8_N, sizeof(int32_t)*_num__array_synapses_8_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_8_N;
    outfile__array_synapses_8_N.open(results_dir + "_array_synapses_8_N_1716210144", ios::binary | ios::out);
    if(outfile__array_synapses_8_N.is_open())
    {
        outfile__array_synapses_8_N.write(reinterpret_cast<char*>(_array_synapses_8_N), 1*sizeof(int32_t));
        outfile__array_synapses_8_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_8_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_9_N, dev_array_synapses_9_N, sizeof(int32_t)*_num__array_synapses_9_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_9_N;
    outfile__array_synapses_9_N.open(results_dir + "_array_synapses_9_N_1737040855", ios::binary | ios::out);
    if(outfile__array_synapses_9_N.is_open())
    {
        outfile__array_synapses_9_N.write(reinterpret_cast<char*>(_array_synapses_9_N), 1*sizeof(int32_t));
        outfile__array_synapses_9_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_9_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_N, dev_array_synapses_N, sizeof(int32_t)*_num__array_synapses_N, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(int32_t));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_sources, dev_array_synapses_sources, sizeof(int32_t)*_num__array_synapses_sources, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_sources;
    outfile__array_synapses_sources.open(results_dir + "_array_synapses_sources_4052383611", ios::binary | ios::out);
    if(outfile__array_synapses_sources.is_open())
    {
        outfile__array_synapses_sources.write(reinterpret_cast<char*>(_array_synapses_sources), 27*sizeof(int32_t));
        outfile__array_synapses_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_sources." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_sources_1, dev_array_synapses_sources_1, sizeof(int32_t)*_num__array_synapses_sources_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_sources_1;
    outfile__array_synapses_sources_1.open(results_dir + "_array_synapses_sources_1_797894262", ios::binary | ios::out);
    if(outfile__array_synapses_sources_1.is_open())
    {
        outfile__array_synapses_sources_1.write(reinterpret_cast<char*>(_array_synapses_sources_1), 243*sizeof(int32_t));
        outfile__array_synapses_sources_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_sources_1." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_targets, dev_array_synapses_targets, sizeof(int32_t)*_num__array_synapses_targets, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_targets;
    outfile__array_synapses_targets.open(results_dir + "_array_synapses_targets_2358512794", ios::binary | ios::out);
    if(outfile__array_synapses_targets.is_open())
    {
        outfile__array_synapses_targets.write(reinterpret_cast<char*>(_array_synapses_targets), 27*sizeof(int32_t));
        outfile__array_synapses_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_targets." << endl;
    }
    CUDA_SAFE_CALL(
            cudaMemcpy(_array_synapses_targets_1, dev_array_synapses_targets_1, sizeof(int32_t)*_num__array_synapses_targets_1, cudaMemcpyDeviceToHost)
            );
    ofstream outfile__array_synapses_targets_1;
    outfile__array_synapses_targets_1.open(results_dir + "_array_synapses_targets_1_2940006769", ios::binary | ios::out);
    if(outfile__array_synapses_targets_1.is_open())
    {
        outfile__array_synapses_targets_1.write(reinterpret_cast<char*>(_array_synapses_targets_1), 243*sizeof(int32_t));
        outfile__array_synapses_targets_1.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_targets_1." << endl;
    }

    _dynamic_array_spikemonitor_i = dev_dynamic_array_spikemonitor_i;
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open(results_dir + "_dynamic_array_spikemonitor_i_1976709050", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[0])), _dynamic_array_spikemonitor_i.size()*sizeof(int32_t));
        outfile__dynamic_array_spikemonitor_i.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    _dynamic_array_spikemonitor_t = dev_dynamic_array_spikemonitor_t;
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open(results_dir + "_dynamic_array_spikemonitor_t_383009635", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[0])), _dynamic_array_spikemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_spikemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_t;
    outfile__dynamic_array_statemonitor_t.open(results_dir + "_dynamic_array_statemonitor_t_3983503110", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_t.is_open())
    {
        outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_t[0])), _dynamic_array_statemonitor_t.size()*sizeof(double));
        outfile__dynamic_array_statemonitor_t.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_10__synaptic_post;
    outfile__dynamic_array_synapses_10__synaptic_post.open(results_dir + "_dynamic_array_synapses_10__synaptic_post_694780356", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_10__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10__synaptic_post[0])), _dynamic_array_synapses_10__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_10__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_10__synaptic_pre;
    outfile__dynamic_array_synapses_10__synaptic_pre.open(results_dir + "_dynamic_array_synapses_10__synaptic_pre_3030054246", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_10__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10__synaptic_pre[0])), _dynamic_array_synapses_10__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_10__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_10_delay;
    outfile__dynamic_array_synapses_10_delay.open(results_dir + "_dynamic_array_synapses_10_delay_2144790254", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10_delay.is_open())
    {
        outfile__dynamic_array_synapses_10_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10_delay[0])), _dynamic_array_synapses_10_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_10_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10_delay." << endl;
    }
    _dynamic_array_synapses_10_N_incoming = dev_dynamic_array_synapses_10_N_incoming;
    ofstream outfile__dynamic_array_synapses_10_N_incoming;
    outfile__dynamic_array_synapses_10_N_incoming.open(results_dir + "_dynamic_array_synapses_10_N_incoming_3245651942", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_10_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10_N_incoming[0])), _dynamic_array_synapses_10_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_10_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10_N_incoming." << endl;
    }
    _dynamic_array_synapses_10_N_outgoing = dev_dynamic_array_synapses_10_N_outgoing;
    ofstream outfile__dynamic_array_synapses_10_N_outgoing;
    outfile__dynamic_array_synapses_10_N_outgoing.open(results_dir + "_dynamic_array_synapses_10_N_outgoing_3865626428", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_10_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10_N_outgoing[0])), _dynamic_array_synapses_10_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_10_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10_N_outgoing." << endl;
    }
    _dynamic_array_synapses_10_w = dev_dynamic_array_synapses_10_w;
    ofstream outfile__dynamic_array_synapses_10_w;
    outfile__dynamic_array_synapses_10_w.open(results_dir + "_dynamic_array_synapses_10_w_1893531083", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_10_w.is_open())
    {
        outfile__dynamic_array_synapses_10_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_10_w[0])), _dynamic_array_synapses_10_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_10_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_10_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_11__synaptic_post;
    outfile__dynamic_array_synapses_11__synaptic_post.open(results_dir + "_dynamic_array_synapses_11__synaptic_post_2264993877", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_11__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11__synaptic_post[0])), _dynamic_array_synapses_11__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_11__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_11__synaptic_pre;
    outfile__dynamic_array_synapses_11__synaptic_pre.open(results_dir + "_dynamic_array_synapses_11__synaptic_pre_224472718", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_11__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11__synaptic_pre[0])), _dynamic_array_synapses_11__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_11__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_11_delay;
    outfile__dynamic_array_synapses_11_delay.open(results_dir + "_dynamic_array_synapses_11_delay_3651267930", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11_delay.is_open())
    {
        outfile__dynamic_array_synapses_11_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11_delay[0])), _dynamic_array_synapses_11_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_11_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11_delay." << endl;
    }
    _dynamic_array_synapses_11_N_incoming = dev_dynamic_array_synapses_11_N_incoming;
    ofstream outfile__dynamic_array_synapses_11_N_incoming;
    outfile__dynamic_array_synapses_11_N_incoming.open(results_dir + "_dynamic_array_synapses_11_N_incoming_1523710857", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_11_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11_N_incoming[0])), _dynamic_array_synapses_11_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_11_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11_N_incoming." << endl;
    }
    _dynamic_array_synapses_11_N_outgoing = dev_dynamic_array_synapses_11_N_outgoing;
    ofstream outfile__dynamic_array_synapses_11_N_outgoing;
    outfile__dynamic_array_synapses_11_N_outgoing.open(results_dir + "_dynamic_array_synapses_11_N_outgoing_2110622547", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_11_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11_N_outgoing[0])), _dynamic_array_synapses_11_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_11_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11_N_outgoing." << endl;
    }
    _dynamic_array_synapses_11_w = dev_dynamic_array_synapses_11_w;
    ofstream outfile__dynamic_array_synapses_11_w;
    outfile__dynamic_array_synapses_11_w.open(results_dir + "_dynamic_array_synapses_11_w_1897830396", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_11_w.is_open())
    {
        outfile__dynamic_array_synapses_11_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_11_w[0])), _dynamic_array_synapses_11_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_11_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_11_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_12__synaptic_post;
    outfile__dynamic_array_synapses_12__synaptic_post.open(results_dir + "_dynamic_array_synapses_12__synaptic_post_2932450471", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_12__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12__synaptic_post[0])), _dynamic_array_synapses_12__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_12__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_12__synaptic_pre;
    outfile__dynamic_array_synapses_12__synaptic_pre.open(results_dir + "_dynamic_array_synapses_12__synaptic_pre_471614711", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_12__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12__synaptic_pre[0])), _dynamic_array_synapses_12__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_12__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_12_delay;
    outfile__dynamic_array_synapses_12_delay.open(results_dir + "_dynamic_array_synapses_12_delay_3897161671", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12_delay.is_open())
    {
        outfile__dynamic_array_synapses_12_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12_delay[0])), _dynamic_array_synapses_12_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_12_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12_delay." << endl;
    }
    _dynamic_array_synapses_12_N_incoming = dev_dynamic_array_synapses_12_N_incoming;
    ofstream outfile__dynamic_array_synapses_12_N_incoming;
    outfile__dynamic_array_synapses_12_N_incoming.open(results_dir + "_dynamic_array_synapses_12_N_incoming_760162681", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_12_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12_N_incoming[0])), _dynamic_array_synapses_12_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_12_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12_N_incoming." << endl;
    }
    _dynamic_array_synapses_12_N_outgoing = dev_dynamic_array_synapses_12_N_outgoing;
    ofstream outfile__dynamic_array_synapses_12_N_outgoing;
    outfile__dynamic_array_synapses_12_N_outgoing.open(results_dir + "_dynamic_array_synapses_12_N_outgoing_173234595", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_12_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12_N_outgoing[0])), _dynamic_array_synapses_12_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_12_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12_N_outgoing." << endl;
    }
    _dynamic_array_synapses_12_w = dev_dynamic_array_synapses_12_w;
    ofstream outfile__dynamic_array_synapses_12_w;
    outfile__dynamic_array_synapses_12_w.open(results_dir + "_dynamic_array_synapses_12_w_1935157669", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_12_w.is_open())
    {
        outfile__dynamic_array_synapses_12_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_12_w[0])), _dynamic_array_synapses_12_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_12_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_12_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_13__synaptic_post;
    outfile__dynamic_array_synapses_13__synaptic_post.open(results_dir + "_dynamic_array_synapses_13__synaptic_post_10564918", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_13__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13__synaptic_post[0])), _dynamic_array_synapses_13__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_13__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_13__synaptic_pre;
    outfile__dynamic_array_synapses_13__synaptic_pre.open(results_dir + "_dynamic_array_synapses_13__synaptic_pre_2783420191", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_13__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13__synaptic_pre[0])), _dynamic_array_synapses_13__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_13__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_13_delay;
    outfile__dynamic_array_synapses_13_delay.open(results_dir + "_dynamic_array_synapses_13_delay_1312750707", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13_delay.is_open())
    {
        outfile__dynamic_array_synapses_13_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13_delay[0])), _dynamic_array_synapses_13_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_13_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13_delay." << endl;
    }
    _dynamic_array_synapses_13_N_incoming = dev_dynamic_array_synapses_13_N_incoming;
    ofstream outfile__dynamic_array_synapses_13_N_incoming;
    outfile__dynamic_array_synapses_13_N_incoming.open(results_dir + "_dynamic_array_synapses_13_N_incoming_3068814614", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_13_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13_N_incoming[0])), _dynamic_array_synapses_13_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_13_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13_N_incoming." << endl;
    }
    _dynamic_array_synapses_13_N_outgoing = dev_dynamic_array_synapses_13_N_outgoing;
    ofstream outfile__dynamic_array_synapses_13_N_outgoing;
    outfile__dynamic_array_synapses_13_N_outgoing.open(results_dir + "_dynamic_array_synapses_13_N_outgoing_2448823756", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_13_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13_N_outgoing[0])), _dynamic_array_synapses_13_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_13_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13_N_outgoing." << endl;
    }
    _dynamic_array_synapses_13_w = dev_dynamic_array_synapses_13_w;
    ofstream outfile__dynamic_array_synapses_13_w;
    outfile__dynamic_array_synapses_13_w.open(results_dir + "_dynamic_array_synapses_13_w_1922712466", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_13_w.is_open())
    {
        outfile__dynamic_array_synapses_13_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_13_w[0])), _dynamic_array_synapses_13_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_13_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_13_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_14__synaptic_post;
    outfile__dynamic_array_synapses_14__synaptic_post.open(results_dir + "_dynamic_array_synapses_14__synaptic_post_4250455363", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_14__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14__synaptic_post[0])), _dynamic_array_synapses_14__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_14__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_14__synaptic_pre;
    outfile__dynamic_array_synapses_14__synaptic_pre.open(results_dir + "_dynamic_array_synapses_14__synaptic_pre_1055297541", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_14__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14__synaptic_pre[0])), _dynamic_array_synapses_14__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_14__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_14_delay;
    outfile__dynamic_array_synapses_14_delay.open(results_dir + "_dynamic_array_synapses_14_delay_2342109949", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14_delay.is_open())
    {
        outfile__dynamic_array_synapses_14_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14_delay[0])), _dynamic_array_synapses_14_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_14_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14_delay." << endl;
    }
    _dynamic_array_synapses_14_N_incoming = dev_dynamic_array_synapses_14_N_incoming;
    ofstream outfile__dynamic_array_synapses_14_N_incoming;
    outfile__dynamic_array_synapses_14_N_incoming.open(results_dir + "_dynamic_array_synapses_14_N_incoming_3262287001", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_14_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14_N_incoming[0])), _dynamic_array_synapses_14_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_14_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14_N_incoming." << endl;
    }
    _dynamic_array_synapses_14_N_outgoing = dev_dynamic_array_synapses_14_N_outgoing;
    ofstream outfile__dynamic_array_synapses_14_N_outgoing;
    outfile__dynamic_array_synapses_14_N_outgoing.open(results_dir + "_dynamic_array_synapses_14_N_outgoing_3849256003", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_14_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14_N_outgoing[0])), _dynamic_array_synapses_14_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_14_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14_N_outgoing." << endl;
    }
    _dynamic_array_synapses_14_w = dev_dynamic_array_synapses_14_w;
    ofstream outfile__dynamic_array_synapses_14_w;
    outfile__dynamic_array_synapses_14_w.open(results_dir + "_dynamic_array_synapses_14_w_2010469655", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_14_w.is_open())
    {
        outfile__dynamic_array_synapses_14_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_14_w[0])), _dynamic_array_synapses_14_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_14_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_14_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_15__synaptic_post;
    outfile__dynamic_array_synapses_15__synaptic_post.open(results_dir + "_dynamic_array_synapses_15__synaptic_post_1395678418", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_15__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15__synaptic_post[0])), _dynamic_array_synapses_15__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_15__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_15__synaptic_pre;
    outfile__dynamic_array_synapses_15__synaptic_pre.open(results_dir + "_dynamic_array_synapses_15__synaptic_pre_2266844141", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_15__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15__synaptic_pre[0])), _dynamic_array_synapses_15__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_15__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_15_delay;
    outfile__dynamic_array_synapses_15_delay.open(results_dir + "_dynamic_array_synapses_15_delay_770623817", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15_delay.is_open())
    {
        outfile__dynamic_array_synapses_15_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15_delay[0])), _dynamic_array_synapses_15_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_15_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15_delay." << endl;
    }
    _dynamic_array_synapses_15_N_incoming = dev_dynamic_array_synapses_15_N_incoming;
    ofstream outfile__dynamic_array_synapses_15_N_incoming;
    outfile__dynamic_array_synapses_15_N_incoming.open(results_dir + "_dynamic_array_synapses_15_N_incoming_1507311862", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_15_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15_N_incoming[0])), _dynamic_array_synapses_15_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_15_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15_N_incoming." << endl;
    }
    _dynamic_array_synapses_15_N_outgoing = dev_dynamic_array_synapses_15_N_outgoing;
    ofstream outfile__dynamic_array_synapses_15_N_outgoing;
    outfile__dynamic_array_synapses_15_N_outgoing.open(results_dir + "_dynamic_array_synapses_15_N_outgoing_2127278124", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_15_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15_N_outgoing[0])), _dynamic_array_synapses_15_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_15_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15_N_outgoing." << endl;
    }
    _dynamic_array_synapses_15_w = dev_dynamic_array_synapses_15_w;
    ofstream outfile__dynamic_array_synapses_15_w;
    outfile__dynamic_array_synapses_15_w.open(results_dir + "_dynamic_array_synapses_15_w_1981234976", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_15_w.is_open())
    {
        outfile__dynamic_array_synapses_15_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_15_w[0])), _dynamic_array_synapses_15_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_15_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_15_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open(results_dir + "_dynamic_array_synapses_1__synaptic_post_1999337987", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_post[0])), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open(results_dir + "_dynamic_array_synapses_1__synaptic_pre_681065502", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1__synaptic_pre[0])), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open(results_dir + "_dynamic_array_synapses_1_delay_2373823482", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_delay[0])), _dynamic_array_synapses_1_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    _dynamic_array_synapses_1_N_incoming = dev_dynamic_array_synapses_1_N_incoming;
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open(results_dir + "_dynamic_array_synapses_1_N_incoming_3469555706", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_incoming[0])), _dynamic_array_synapses_1_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    _dynamic_array_synapses_1_N_outgoing = dev_dynamic_array_synapses_1_N_outgoing;
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open(results_dir + "_dynamic_array_synapses_1_N_outgoing_3922806560", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_N_outgoing[0])), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_1_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    _dynamic_array_synapses_1_w = dev_dynamic_array_synapses_1_w;
    ofstream outfile__dynamic_array_synapses_1_w;
    outfile__dynamic_array_synapses_1_w.open(results_dir + "_dynamic_array_synapses_1_w_1857285062", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_w.is_open())
    {
        outfile__dynamic_array_synapses_1_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_1_w[0])), _dynamic_array_synapses_1_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_1_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open(results_dir + "_dynamic_array_synapses_2__synaptic_post_1591987953", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_post[0])), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open(results_dir + "_dynamic_array_synapses_2__synaptic_pre_971331175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2__synaptic_pre[0])), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open(results_dir + "_dynamic_array_synapses_2_delay_3163926887", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_delay[0])), _dynamic_array_synapses_2_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    _dynamic_array_synapses_2_N_incoming = dev_dynamic_array_synapses_2_N_incoming;
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open(results_dir + "_dynamic_array_synapses_2_N_incoming_3109283082", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_incoming[0])), _dynamic_array_synapses_2_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    _dynamic_array_synapses_2_N_outgoing = dev_dynamic_array_synapses_2_N_outgoing;
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open(results_dir + "_dynamic_array_synapses_2_N_outgoing_2656015824", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_N_outgoing[0])), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_2_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    _dynamic_array_synapses_2_w = dev_dynamic_array_synapses_2_w;
    ofstream outfile__dynamic_array_synapses_2_w;
    outfile__dynamic_array_synapses_2_w.open(results_dir + "_dynamic_array_synapses_2_w_1828017567", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_w.is_open())
    {
        outfile__dynamic_array_synapses_2_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_2_w[0])), _dynamic_array_synapses_2_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_2_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_post;
    outfile__dynamic_array_synapses_3__synaptic_post.open(results_dir + "_dynamic_array_synapses_3__synaptic_post_4035665760", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_post[0])), _dynamic_array_synapses_3__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_pre;
    outfile__dynamic_array_synapses_3__synaptic_pre.open(results_dir + "_dynamic_array_synapses_3__synaptic_pre_2149485967", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_3__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3__synaptic_pre[0])), _dynamic_array_synapses_3__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_delay;
    outfile__dynamic_array_synapses_3_delay.open(results_dir + "_dynamic_array_synapses_3_delay_451066579", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_delay.is_open())
    {
        outfile__dynamic_array_synapses_3_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_delay[0])), _dynamic_array_synapses_3_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_delay." << endl;
    }
    _dynamic_array_synapses_3_N_incoming = dev_dynamic_array_synapses_3_N_incoming;
    ofstream outfile__dynamic_array_synapses_3_N_incoming;
    outfile__dynamic_array_synapses_3_N_incoming.open(results_dir + "_dynamic_array_synapses_3_N_incoming_586590565", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_3_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_incoming[0])), _dynamic_array_synapses_3_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_incoming." << endl;
    }
    _dynamic_array_synapses_3_N_outgoing = dev_dynamic_array_synapses_3_N_outgoing;
    ofstream outfile__dynamic_array_synapses_3_N_outgoing;
    outfile__dynamic_array_synapses_3_N_outgoing.open(results_dir + "_dynamic_array_synapses_3_N_outgoing_99277247", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_3_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_N_outgoing[0])), _dynamic_array_synapses_3_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_3_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_outgoing." << endl;
    }
    _dynamic_array_synapses_3_w = dev_dynamic_array_synapses_3_w;
    ofstream outfile__dynamic_array_synapses_3_w;
    outfile__dynamic_array_synapses_3_w.open(results_dir + "_dynamic_array_synapses_3_w_1832337320", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_w.is_open())
    {
        outfile__dynamic_array_synapses_3_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_3_w[0])), _dynamic_array_synapses_3_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_3_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_post;
    outfile__dynamic_array_synapses_4__synaptic_post.open(results_dir + "_dynamic_array_synapses_4__synaptic_post_225617685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_4__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4__synaptic_post[0])), _dynamic_array_synapses_4__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_pre;
    outfile__dynamic_array_synapses_4__synaptic_pre.open(results_dir + "_dynamic_array_synapses_4__synaptic_pre_455049877", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_4__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4__synaptic_pre[0])), _dynamic_array_synapses_4__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_delay;
    outfile__dynamic_array_synapses_4_delay.open(results_dir + "_dynamic_array_synapses_4_delay_3745875037", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_delay.is_open())
    {
        outfile__dynamic_array_synapses_4_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_delay[0])), _dynamic_array_synapses_4_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_4_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_delay." << endl;
    }
    _dynamic_array_synapses_4_N_incoming = dev_dynamic_array_synapses_4_N_incoming;
    ofstream outfile__dynamic_array_synapses_4_N_incoming;
    outfile__dynamic_array_synapses_4_N_incoming.open(results_dir + "_dynamic_array_synapses_4_N_incoming_1450066154", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_4_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_N_incoming[0])), _dynamic_array_synapses_4_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_incoming." << endl;
    }
    _dynamic_array_synapses_4_N_outgoing = dev_dynamic_array_synapses_4_N_outgoing;
    ofstream outfile__dynamic_array_synapses_4_N_outgoing;
    outfile__dynamic_array_synapses_4_N_outgoing.open(results_dir + "_dynamic_array_synapses_4_N_outgoing_1903308848", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_4_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_N_outgoing[0])), _dynamic_array_synapses_4_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_4_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_outgoing." << endl;
    }
    _dynamic_array_synapses_4_w = dev_dynamic_array_synapses_4_w;
    ofstream outfile__dynamic_array_synapses_4_w;
    outfile__dynamic_array_synapses_4_w.open(results_dir + "_dynamic_array_synapses_4_w_1752705325", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_w.is_open())
    {
        outfile__dynamic_array_synapses_4_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_4_w[0])), _dynamic_array_synapses_4_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_4_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_post;
    outfile__dynamic_array_synapses_5__synaptic_post.open(results_dir + "_dynamic_array_synapses_5__synaptic_post_2736404100", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_5__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5__synaptic_post[0])), _dynamic_array_synapses_5__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_pre;
    outfile__dynamic_array_synapses_5__synaptic_pre.open(results_dir + "_dynamic_array_synapses_5__synaptic_pre_2732874109", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_5__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5__synaptic_pre[0])), _dynamic_array_synapses_5__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_delay;
    outfile__dynamic_array_synapses_5_delay.open(results_dir + "_dynamic_array_synapses_5_delay_2033356777", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_delay.is_open())
    {
        outfile__dynamic_array_synapses_5_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_delay[0])), _dynamic_array_synapses_5_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_5_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_delay." << endl;
    }
    _dynamic_array_synapses_5_N_incoming = dev_dynamic_array_synapses_5_N_incoming;
    ofstream outfile__dynamic_array_synapses_5_N_incoming;
    outfile__dynamic_array_synapses_5_N_incoming.open(results_dir + "_dynamic_array_synapses_5_N_incoming_3452636293", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_5_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_N_incoming[0])), _dynamic_array_synapses_5_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_incoming." << endl;
    }
    _dynamic_array_synapses_5_N_outgoing = dev_dynamic_array_synapses_5_N_outgoing;
    ofstream outfile__dynamic_array_synapses_5_N_outgoing;
    outfile__dynamic_array_synapses_5_N_outgoing.open(results_dir + "_dynamic_array_synapses_5_N_outgoing_3939990623", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_5_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_N_outgoing[0])), _dynamic_array_synapses_5_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_5_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_outgoing." << endl;
    }
    _dynamic_array_synapses_5_w = dev_dynamic_array_synapses_5_w;
    ofstream outfile__dynamic_array_synapses_5_w;
    outfile__dynamic_array_synapses_5_w.open(results_dir + "_dynamic_array_synapses_5_w_1773814554", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_w.is_open())
    {
        outfile__dynamic_array_synapses_5_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_5_w[0])), _dynamic_array_synapses_5_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_5_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_post;
    outfile__dynamic_array_synapses_6__synaptic_post.open(results_dir + "_dynamic_array_synapses_6__synaptic_post_2329051766", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_6__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6__synaptic_post[0])), _dynamic_array_synapses_6__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_pre;
    outfile__dynamic_array_synapses_6__synaptic_pre.open(results_dir + "_dynamic_array_synapses_6__synaptic_pre_3013161732", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_6__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6__synaptic_pre[0])), _dynamic_array_synapses_6__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_delay;
    outfile__dynamic_array_synapses_6_delay.open(results_dir + "_dynamic_array_synapses_6_delay_1222284660", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_delay.is_open())
    {
        outfile__dynamic_array_synapses_6_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_delay[0])), _dynamic_array_synapses_6_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_6_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_delay." << endl;
    }
    _dynamic_array_synapses_6_N_incoming = dev_dynamic_array_synapses_6_N_incoming;
    ofstream outfile__dynamic_array_synapses_6_N_incoming;
    outfile__dynamic_array_synapses_6_N_incoming.open(results_dir + "_dynamic_array_synapses_6_N_incoming_3126189685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_6_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_N_incoming[0])), _dynamic_array_synapses_6_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_incoming." << endl;
    }
    _dynamic_array_synapses_6_N_outgoing = dev_dynamic_array_synapses_6_N_outgoing;
    ofstream outfile__dynamic_array_synapses_6_N_outgoing;
    outfile__dynamic_array_synapses_6_N_outgoing.open(results_dir + "_dynamic_array_synapses_6_N_outgoing_2638851759", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_6_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_N_outgoing[0])), _dynamic_array_synapses_6_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_6_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_outgoing." << endl;
    }
    _dynamic_array_synapses_6_w = dev_dynamic_array_synapses_6_w;
    ofstream outfile__dynamic_array_synapses_6_w;
    outfile__dynamic_array_synapses_6_w.open(results_dir + "_dynamic_array_synapses_6_w_1811742019", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_w.is_open())
    {
        outfile__dynamic_array_synapses_6_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_6_w[0])), _dynamic_array_synapses_6_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_6_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_7__synaptic_post;
    outfile__dynamic_array_synapses_7__synaptic_post.open(results_dir + "_dynamic_array_synapses_7__synaptic_post_616174567", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_7__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7__synaptic_post[0])), _dynamic_array_synapses_7__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_7__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_7__synaptic_pre;
    outfile__dynamic_array_synapses_7__synaptic_pre.open(results_dir + "_dynamic_array_synapses_7__synaptic_pre_174254316", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_7__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7__synaptic_pre[0])), _dynamic_array_synapses_7__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_7__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_7_delay;
    outfile__dynamic_array_synapses_7_delay.open(results_dir + "_dynamic_array_synapses_7_delay_4004355776", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7_delay.is_open())
    {
        outfile__dynamic_array_synapses_7_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7_delay[0])), _dynamic_array_synapses_7_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_7_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7_delay." << endl;
    }
    _dynamic_array_synapses_7_N_incoming = dev_dynamic_array_synapses_7_N_incoming;
    ofstream outfile__dynamic_array_synapses_7_N_incoming;
    outfile__dynamic_array_synapses_7_N_incoming.open(results_dir + "_dynamic_array_synapses_7_N_incoming_569414170", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_7_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7_N_incoming[0])), _dynamic_array_synapses_7_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_7_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7_N_incoming." << endl;
    }
    _dynamic_array_synapses_7_N_outgoing = dev_dynamic_array_synapses_7_N_outgoing;
    ofstream outfile__dynamic_array_synapses_7_N_outgoing;
    outfile__dynamic_array_synapses_7_N_outgoing.open(results_dir + "_dynamic_array_synapses_7_N_outgoing_116187840", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_7_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7_N_outgoing[0])), _dynamic_array_synapses_7_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_7_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7_N_outgoing." << endl;
    }
    _dynamic_array_synapses_7_w = dev_dynamic_array_synapses_7_w;
    ofstream outfile__dynamic_array_synapses_7_w;
    outfile__dynamic_array_synapses_7_w.open(results_dir + "_dynamic_array_synapses_7_w_1782486900", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_7_w.is_open())
    {
        outfile__dynamic_array_synapses_7_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_7_w[0])), _dynamic_array_synapses_7_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_7_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_7_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_8__synaptic_post;
    outfile__dynamic_array_synapses_8__synaptic_post.open(results_dir + "_dynamic_array_synapses_8__synaptic_post_2857399517", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_8__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8__synaptic_post[0])), _dynamic_array_synapses_8__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_8__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_8__synaptic_pre;
    outfile__dynamic_array_synapses_8__synaptic_pre.open(results_dir + "_dynamic_array_synapses_8__synaptic_pre_1592404849", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_8__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8__synaptic_pre[0])), _dynamic_array_synapses_8__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_8__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_8_delay;
    outfile__dynamic_array_synapses_8_delay.open(results_dir + "_dynamic_array_synapses_8_delay_417721897", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8_delay.is_open())
    {
        outfile__dynamic_array_synapses_8_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8_delay[0])), _dynamic_array_synapses_8_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_8_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8_delay." << endl;
    }
    _dynamic_array_synapses_8_N_incoming = dev_dynamic_array_synapses_8_N_incoming;
    ofstream outfile__dynamic_array_synapses_8_N_incoming;
    outfile__dynamic_array_synapses_8_N_incoming.open(results_dir + "_dynamic_array_synapses_8_N_incoming_1399065963", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_8_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8_N_incoming[0])), _dynamic_array_synapses_8_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_8_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8_N_incoming." << endl;
    }
    _dynamic_array_synapses_8_N_outgoing = dev_dynamic_array_synapses_8_N_outgoing;
    ofstream outfile__dynamic_array_synapses_8_N_outgoing;
    outfile__dynamic_array_synapses_8_N_outgoing.open(results_dir + "_dynamic_array_synapses_8_N_outgoing_1954053553", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_8_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8_N_outgoing[0])), _dynamic_array_synapses_8_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_8_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8_N_outgoing." << endl;
    }
    _dynamic_array_synapses_8_w = dev_dynamic_array_synapses_8_w;
    ofstream outfile__dynamic_array_synapses_8_w;
    outfile__dynamic_array_synapses_8_w.open(results_dir + "_dynamic_array_synapses_8_w_1633865801", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_8_w.is_open())
    {
        outfile__dynamic_array_synapses_8_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_8_w[0])), _dynamic_array_synapses_8_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_8_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_8_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses_9__synaptic_post;
    outfile__dynamic_array_synapses_9__synaptic_post.open(results_dir + "_dynamic_array_synapses_9__synaptic_post_70837580", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses_9__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9__synaptic_post[0])), _dynamic_array_synapses_9__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_9__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_9__synaptic_pre;
    outfile__dynamic_array_synapses_9__synaptic_pre.open(results_dir + "_dynamic_array_synapses_9__synaptic_pre_3876712601", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses_9__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9__synaptic_pre[0])), _dynamic_array_synapses_9__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_9__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_9_delay;
    outfile__dynamic_array_synapses_9_delay.open(results_dir + "_dynamic_array_synapses_9_delay_3197298077", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9_delay.is_open())
    {
        outfile__dynamic_array_synapses_9_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9_delay[0])), _dynamic_array_synapses_9_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_9_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9_delay." << endl;
    }
    _dynamic_array_synapses_9_N_incoming = dev_dynamic_array_synapses_9_N_incoming;
    ofstream outfile__dynamic_array_synapses_9_N_incoming;
    outfile__dynamic_array_synapses_9_N_incoming.open(results_dir + "_dynamic_array_synapses_9_N_incoming_3368108292", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_9_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9_N_incoming[0])), _dynamic_array_synapses_9_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_9_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9_N_incoming." << endl;
    }
    _dynamic_array_synapses_9_N_outgoing = dev_dynamic_array_synapses_9_N_outgoing;
    ofstream outfile__dynamic_array_synapses_9_N_outgoing;
    outfile__dynamic_array_synapses_9_N_outgoing.open(results_dir + "_dynamic_array_synapses_9_N_outgoing_4024250846", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_9_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9_N_outgoing[0])), _dynamic_array_synapses_9_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_9_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9_N_outgoing." << endl;
    }
    _dynamic_array_synapses_9_w = dev_dynamic_array_synapses_9_w;
    ofstream outfile__dynamic_array_synapses_9_w;
    outfile__dynamic_array_synapses_9_w.open(results_dir + "_dynamic_array_synapses_9_w_1621146238", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_9_w.is_open())
    {
        outfile__dynamic_array_synapses_9_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_9_w[0])), _dynamic_array_synapses_9_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_9_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_9_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0])), _dynamic_array_synapses__synaptic_post.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_post.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0])), _dynamic_array_synapses__synaptic_pre.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses__synaptic_pre.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_delay[0])), _dynamic_array_synapses_delay.size()*sizeof(double));
        outfile__dynamic_array_synapses_delay.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    _dynamic_array_synapses_N_incoming = dev_dynamic_array_synapses_N_incoming;
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_incoming[0])), _dynamic_array_synapses_N_incoming.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_incoming.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    _dynamic_array_synapses_N_outgoing = dev_dynamic_array_synapses_N_outgoing;
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_N_outgoing[0])), _dynamic_array_synapses_N_outgoing.size()*sizeof(int32_t));
        outfile__dynamic_array_synapses_N_outgoing.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    _dynamic_array_synapses_w = dev_dynamic_array_synapses_w;
    ofstream outfile__dynamic_array_synapses_w;
    outfile__dynamic_array_synapses_w.open(results_dir + "_dynamic_array_synapses_w_441891901", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_w.is_open())
    {
        outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0])), _dynamic_array_synapses_w.size()*sizeof(double));
        outfile__dynamic_array_synapses_w.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
    }

        ofstream outfile__dynamic_array_statemonitor_I;
        outfile__dynamic_array_statemonitor_I.open(results_dir + "_dynamic_array_statemonitor_I_3037143831", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_I.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_I = new thrust::host_vector<double>[_num__array_statemonitor__indices];
            for (int n=0; n<_num__array_statemonitor__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_I[n] = _dynamic_array_statemonitor_I[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_I[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_I.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_I[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_I.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_I." << endl;
        }
        ofstream outfile__dynamic_array_statemonitor_v;
        outfile__dynamic_array_statemonitor_v.open(results_dir + "_dynamic_array_statemonitor_v_56692266", ios::binary | ios::out);
        if(outfile__dynamic_array_statemonitor_v.is_open())
        {
            thrust::host_vector<double>* temp_array_dynamic_array_statemonitor_v = new thrust::host_vector<double>[_num__array_statemonitor__indices];
            for (int n=0; n<_num__array_statemonitor__indices; n++)
            {
                temp_array_dynamic_array_statemonitor_v[n] = _dynamic_array_statemonitor_v[n];
            }
            for(int j = 0; j < temp_array_dynamic_array_statemonitor_v[0].size(); j++)
            {
                for(int i = 0; i < _num__array_statemonitor__indices; i++)
                {
                    outfile__dynamic_array_statemonitor_v.write(reinterpret_cast<char*>(&temp_array_dynamic_array_statemonitor_v[i][j]), sizeof(double));
                }
            }
            outfile__dynamic_array_statemonitor_v.close();
        } else
        {
            std::cout << "Error writing output file for _dynamic_array_statemonitor_v." << endl;
        }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

__global__ void synapses_pre_destroy()
{
    using namespace brian;

    synapses_pre.destroy();
}
__global__ void synapses_1_pre_destroy()
{
    using namespace brian;

    synapses_1_pre.destroy();
}
__global__ void synapses_10_pre_destroy()
{
    using namespace brian;

    synapses_10_pre.destroy();
}
__global__ void synapses_11_pre_destroy()
{
    using namespace brian;

    synapses_11_pre.destroy();
}
__global__ void synapses_12_pre_destroy()
{
    using namespace brian;

    synapses_12_pre.destroy();
}
__global__ void synapses_13_pre_destroy()
{
    using namespace brian;

    synapses_13_pre.destroy();
}
__global__ void synapses_14_pre_destroy()
{
    using namespace brian;

    synapses_14_pre.destroy();
}
__global__ void synapses_15_pre_destroy()
{
    using namespace brian;

    synapses_15_pre.destroy();
}
__global__ void synapses_2_pre_destroy()
{
    using namespace brian;

    synapses_2_pre.destroy();
}
__global__ void synapses_3_pre_destroy()
{
    using namespace brian;

    synapses_3_pre.destroy();
}
__global__ void synapses_4_pre_destroy()
{
    using namespace brian;

    synapses_4_pre.destroy();
}
__global__ void synapses_5_pre_destroy()
{
    using namespace brian;

    synapses_5_pre.destroy();
}
__global__ void synapses_6_pre_destroy()
{
    using namespace brian;

    synapses_6_pre.destroy();
}
__global__ void synapses_7_pre_destroy()
{
    using namespace brian;

    synapses_7_pre.destroy();
}
__global__ void synapses_8_pre_destroy()
{
    using namespace brian;

    synapses_8_pre.destroy();
}
__global__ void synapses_9_pre_destroy()
{
    using namespace brian;

    synapses_9_pre.destroy();
}

void _dealloc_arrays()
{
    using namespace brian;


    CUDA_SAFE_CALL(
            curandDestroyGenerator(curand_generator)
            );

    synapses_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_pre_destroy");
    synapses_1_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_1_pre_destroy");
    synapses_10_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_10_pre_destroy");
    synapses_11_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_11_pre_destroy");
    synapses_12_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_12_pre_destroy");
    synapses_13_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_13_pre_destroy");
    synapses_14_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_14_pre_destroy");
    synapses_15_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_15_pre_destroy");
    synapses_2_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_2_pre_destroy");
    synapses_3_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_3_pre_destroy");
    synapses_4_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_4_pre_destroy");
    synapses_5_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_5_pre_destroy");
    synapses_6_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_6_pre_destroy");
    synapses_7_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_7_pre_destroy");
    synapses_8_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_8_pre_destroy");
    synapses_9_pre_destroy<<<1,1>>>();
    CUDA_CHECK_ERROR("synapses_9_pre_destroy");

    dev_dynamic_array_spikemonitor_i.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_spikemonitor_i);
    _dynamic_array_spikemonitor_i.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_spikemonitor_i);
    dev_dynamic_array_spikemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_spikemonitor_t);
    _dynamic_array_spikemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_spikemonitor_t);
    dev_dynamic_array_statemonitor_t.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_statemonitor_t);
    _dynamic_array_statemonitor_t.clear();
    thrust::host_vector<double>().swap(_dynamic_array_statemonitor_t);
    dev_dynamic_array_synapses_10__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_10__synaptic_post);
    _dynamic_array_synapses_10__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_10__synaptic_post);
    dev_dynamic_array_synapses_10__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_10__synaptic_pre);
    _dynamic_array_synapses_10__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_10__synaptic_pre);
    dev_dynamic_array_synapses_10_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_10_delay);
    _dynamic_array_synapses_10_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_10_delay);
    dev_dynamic_array_synapses_10_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_10_N_incoming);
    _dynamic_array_synapses_10_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_10_N_incoming);
    dev_dynamic_array_synapses_10_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_10_N_outgoing);
    _dynamic_array_synapses_10_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_10_N_outgoing);
    dev_dynamic_array_synapses_10_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_10_w);
    _dynamic_array_synapses_10_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_10_w);
    dev_dynamic_array_synapses_11__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_11__synaptic_post);
    _dynamic_array_synapses_11__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_11__synaptic_post);
    dev_dynamic_array_synapses_11__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_11__synaptic_pre);
    _dynamic_array_synapses_11__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_11__synaptic_pre);
    dev_dynamic_array_synapses_11_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_11_delay);
    _dynamic_array_synapses_11_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_11_delay);
    dev_dynamic_array_synapses_11_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_11_N_incoming);
    _dynamic_array_synapses_11_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_11_N_incoming);
    dev_dynamic_array_synapses_11_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_11_N_outgoing);
    _dynamic_array_synapses_11_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_11_N_outgoing);
    dev_dynamic_array_synapses_11_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_11_w);
    _dynamic_array_synapses_11_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_11_w);
    dev_dynamic_array_synapses_12__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_12__synaptic_post);
    _dynamic_array_synapses_12__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_12__synaptic_post);
    dev_dynamic_array_synapses_12__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_12__synaptic_pre);
    _dynamic_array_synapses_12__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_12__synaptic_pre);
    dev_dynamic_array_synapses_12_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_12_delay);
    _dynamic_array_synapses_12_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_12_delay);
    dev_dynamic_array_synapses_12_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_12_N_incoming);
    _dynamic_array_synapses_12_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_12_N_incoming);
    dev_dynamic_array_synapses_12_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_12_N_outgoing);
    _dynamic_array_synapses_12_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_12_N_outgoing);
    dev_dynamic_array_synapses_12_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_12_w);
    _dynamic_array_synapses_12_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_12_w);
    dev_dynamic_array_synapses_13__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_13__synaptic_post);
    _dynamic_array_synapses_13__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_13__synaptic_post);
    dev_dynamic_array_synapses_13__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_13__synaptic_pre);
    _dynamic_array_synapses_13__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_13__synaptic_pre);
    dev_dynamic_array_synapses_13_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_13_delay);
    _dynamic_array_synapses_13_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_13_delay);
    dev_dynamic_array_synapses_13_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_13_N_incoming);
    _dynamic_array_synapses_13_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_13_N_incoming);
    dev_dynamic_array_synapses_13_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_13_N_outgoing);
    _dynamic_array_synapses_13_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_13_N_outgoing);
    dev_dynamic_array_synapses_13_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_13_w);
    _dynamic_array_synapses_13_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_13_w);
    dev_dynamic_array_synapses_14__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_14__synaptic_post);
    _dynamic_array_synapses_14__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_14__synaptic_post);
    dev_dynamic_array_synapses_14__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_14__synaptic_pre);
    _dynamic_array_synapses_14__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_14__synaptic_pre);
    dev_dynamic_array_synapses_14_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_14_delay);
    _dynamic_array_synapses_14_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_14_delay);
    dev_dynamic_array_synapses_14_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_14_N_incoming);
    _dynamic_array_synapses_14_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_14_N_incoming);
    dev_dynamic_array_synapses_14_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_14_N_outgoing);
    _dynamic_array_synapses_14_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_14_N_outgoing);
    dev_dynamic_array_synapses_14_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_14_w);
    _dynamic_array_synapses_14_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_14_w);
    dev_dynamic_array_synapses_15__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_15__synaptic_post);
    _dynamic_array_synapses_15__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_15__synaptic_post);
    dev_dynamic_array_synapses_15__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_15__synaptic_pre);
    _dynamic_array_synapses_15__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_15__synaptic_pre);
    dev_dynamic_array_synapses_15_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_15_delay);
    _dynamic_array_synapses_15_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_15_delay);
    dev_dynamic_array_synapses_15_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_15_N_incoming);
    _dynamic_array_synapses_15_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_15_N_incoming);
    dev_dynamic_array_synapses_15_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_15_N_outgoing);
    _dynamic_array_synapses_15_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_15_N_outgoing);
    dev_dynamic_array_synapses_15_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_15_w);
    _dynamic_array_synapses_15_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_15_w);
    dev_dynamic_array_synapses_1__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_post);
    _dynamic_array_synapses_1__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_post);
    dev_dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1__synaptic_pre);
    _dynamic_array_synapses_1__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1__synaptic_pre);
    dev_dynamic_array_synapses_1_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_delay);
    _dynamic_array_synapses_1_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_delay);
    dev_dynamic_array_synapses_1_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_incoming);
    _dynamic_array_synapses_1_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_incoming);
    dev_dynamic_array_synapses_1_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_1_N_outgoing);
    _dynamic_array_synapses_1_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_1_N_outgoing);
    dev_dynamic_array_synapses_1_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_1_w);
    _dynamic_array_synapses_1_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_1_w);
    dev_dynamic_array_synapses_2__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_post);
    _dynamic_array_synapses_2__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_post);
    dev_dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2__synaptic_pre);
    _dynamic_array_synapses_2__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2__synaptic_pre);
    dev_dynamic_array_synapses_2_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_delay);
    _dynamic_array_synapses_2_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_delay);
    dev_dynamic_array_synapses_2_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_incoming);
    _dynamic_array_synapses_2_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_incoming);
    dev_dynamic_array_synapses_2_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_2_N_outgoing);
    _dynamic_array_synapses_2_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_2_N_outgoing);
    dev_dynamic_array_synapses_2_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_2_w);
    _dynamic_array_synapses_2_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_2_w);
    dev_dynamic_array_synapses_3__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_post);
    _dynamic_array_synapses_3__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_post);
    dev_dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3__synaptic_pre);
    _dynamic_array_synapses_3__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3__synaptic_pre);
    dev_dynamic_array_synapses_3_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_delay);
    _dynamic_array_synapses_3_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_delay);
    dev_dynamic_array_synapses_3_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_incoming);
    _dynamic_array_synapses_3_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_incoming);
    dev_dynamic_array_synapses_3_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_3_N_outgoing);
    _dynamic_array_synapses_3_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_3_N_outgoing);
    dev_dynamic_array_synapses_3_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_3_w);
    _dynamic_array_synapses_3_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_3_w);
    dev_dynamic_array_synapses_4__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4__synaptic_post);
    _dynamic_array_synapses_4__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4__synaptic_post);
    dev_dynamic_array_synapses_4__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4__synaptic_pre);
    _dynamic_array_synapses_4__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4__synaptic_pre);
    dev_dynamic_array_synapses_4_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_4_delay);
    _dynamic_array_synapses_4_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_4_delay);
    dev_dynamic_array_synapses_4_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4_N_incoming);
    _dynamic_array_synapses_4_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4_N_incoming);
    dev_dynamic_array_synapses_4_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_4_N_outgoing);
    _dynamic_array_synapses_4_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_4_N_outgoing);
    dev_dynamic_array_synapses_4_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_4_w);
    _dynamic_array_synapses_4_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_4_w);
    dev_dynamic_array_synapses_5__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5__synaptic_post);
    _dynamic_array_synapses_5__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5__synaptic_post);
    dev_dynamic_array_synapses_5__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5__synaptic_pre);
    _dynamic_array_synapses_5__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5__synaptic_pre);
    dev_dynamic_array_synapses_5_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_5_delay);
    _dynamic_array_synapses_5_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_5_delay);
    dev_dynamic_array_synapses_5_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5_N_incoming);
    _dynamic_array_synapses_5_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5_N_incoming);
    dev_dynamic_array_synapses_5_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_5_N_outgoing);
    _dynamic_array_synapses_5_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_5_N_outgoing);
    dev_dynamic_array_synapses_5_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_5_w);
    _dynamic_array_synapses_5_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_5_w);
    dev_dynamic_array_synapses_6__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6__synaptic_post);
    _dynamic_array_synapses_6__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6__synaptic_post);
    dev_dynamic_array_synapses_6__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6__synaptic_pre);
    _dynamic_array_synapses_6__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6__synaptic_pre);
    dev_dynamic_array_synapses_6_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_6_delay);
    _dynamic_array_synapses_6_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_6_delay);
    dev_dynamic_array_synapses_6_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6_N_incoming);
    _dynamic_array_synapses_6_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6_N_incoming);
    dev_dynamic_array_synapses_6_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_6_N_outgoing);
    _dynamic_array_synapses_6_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_6_N_outgoing);
    dev_dynamic_array_synapses_6_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_6_w);
    _dynamic_array_synapses_6_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_6_w);
    dev_dynamic_array_synapses_7__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_7__synaptic_post);
    _dynamic_array_synapses_7__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_7__synaptic_post);
    dev_dynamic_array_synapses_7__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_7__synaptic_pre);
    _dynamic_array_synapses_7__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_7__synaptic_pre);
    dev_dynamic_array_synapses_7_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_7_delay);
    _dynamic_array_synapses_7_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_7_delay);
    dev_dynamic_array_synapses_7_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_7_N_incoming);
    _dynamic_array_synapses_7_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_7_N_incoming);
    dev_dynamic_array_synapses_7_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_7_N_outgoing);
    _dynamic_array_synapses_7_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_7_N_outgoing);
    dev_dynamic_array_synapses_7_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_7_w);
    _dynamic_array_synapses_7_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_7_w);
    dev_dynamic_array_synapses_8__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_8__synaptic_post);
    _dynamic_array_synapses_8__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_8__synaptic_post);
    dev_dynamic_array_synapses_8__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_8__synaptic_pre);
    _dynamic_array_synapses_8__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_8__synaptic_pre);
    dev_dynamic_array_synapses_8_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_8_delay);
    _dynamic_array_synapses_8_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_8_delay);
    dev_dynamic_array_synapses_8_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_8_N_incoming);
    _dynamic_array_synapses_8_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_8_N_incoming);
    dev_dynamic_array_synapses_8_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_8_N_outgoing);
    _dynamic_array_synapses_8_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_8_N_outgoing);
    dev_dynamic_array_synapses_8_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_8_w);
    _dynamic_array_synapses_8_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_8_w);
    dev_dynamic_array_synapses_9__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_9__synaptic_post);
    _dynamic_array_synapses_9__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_9__synaptic_post);
    dev_dynamic_array_synapses_9__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_9__synaptic_pre);
    _dynamic_array_synapses_9__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_9__synaptic_pre);
    dev_dynamic_array_synapses_9_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_9_delay);
    _dynamic_array_synapses_9_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_9_delay);
    dev_dynamic_array_synapses_9_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_9_N_incoming);
    _dynamic_array_synapses_9_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_9_N_incoming);
    dev_dynamic_array_synapses_9_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_9_N_outgoing);
    _dynamic_array_synapses_9_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_9_N_outgoing);
    dev_dynamic_array_synapses_9_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_9_w);
    _dynamic_array_synapses_9_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_9_w);
    dev_dynamic_array_synapses__synaptic_post.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_post);
    _dynamic_array_synapses__synaptic_post.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
    dev_dynamic_array_synapses__synaptic_pre.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses__synaptic_pre);
    _dynamic_array_synapses__synaptic_pre.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
    dev_dynamic_array_synapses_delay.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_delay);
    _dynamic_array_synapses_delay.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_delay);
    dev_dynamic_array_synapses_N_incoming.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_incoming);
    _dynamic_array_synapses_N_incoming.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_incoming);
    dev_dynamic_array_synapses_N_outgoing.clear();
    thrust::device_vector<int32_t>().swap(dev_dynamic_array_synapses_N_outgoing);
    _dynamic_array_synapses_N_outgoing.clear();
    thrust::host_vector<int32_t>().swap(_dynamic_array_synapses_N_outgoing);
    dev_dynamic_array_synapses_w.clear();
    thrust::device_vector<double>().swap(dev_dynamic_array_synapses_w);
    _dynamic_array_synapses_w.clear();
    thrust::host_vector<double>().swap(_dynamic_array_synapses_w);

    if(_array_defaultclock_dt!=0)
    {
        delete [] _array_defaultclock_dt;
        _array_defaultclock_dt = 0;
    }
    if(dev_array_defaultclock_dt!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_dt)
                );
        dev_array_defaultclock_dt = 0;
    }
    if(_array_defaultclock_t!=0)
    {
        delete [] _array_defaultclock_t;
        _array_defaultclock_t = 0;
    }
    if(dev_array_defaultclock_t!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_t)
                );
        dev_array_defaultclock_t = 0;
    }
    if(_array_defaultclock_timestep!=0)
    {
        delete [] _array_defaultclock_timestep;
        _array_defaultclock_timestep = 0;
    }
    if(dev_array_defaultclock_timestep!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_defaultclock_timestep)
                );
        dev_array_defaultclock_timestep = 0;
    }
    if(_array_neurongroup_10_A!=0)
    {
        delete [] _array_neurongroup_10_A;
        _array_neurongroup_10_A = 0;
    }
    if(dev_array_neurongroup_10_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_A)
                );
        dev_array_neurongroup_10_A = 0;
    }
    if(_array_neurongroup_10_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_10_batch_sum_X;
        _array_neurongroup_10_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_10_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_batch_sum_X)
                );
        dev_array_neurongroup_10_batch_sum_X = 0;
    }
    if(_array_neurongroup_10_i!=0)
    {
        delete [] _array_neurongroup_10_i;
        _array_neurongroup_10_i = 0;
    }
    if(dev_array_neurongroup_10_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_i)
                );
        dev_array_neurongroup_10_i = 0;
    }
    if(_array_neurongroup_10_Iexc!=0)
    {
        delete [] _array_neurongroup_10_Iexc;
        _array_neurongroup_10_Iexc = 0;
    }
    if(dev_array_neurongroup_10_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_Iexc)
                );
        dev_array_neurongroup_10_Iexc = 0;
    }
    if(_array_neurongroup_10_Iinh!=0)
    {
        delete [] _array_neurongroup_10_Iinh;
        _array_neurongroup_10_Iinh = 0;
    }
    if(dev_array_neurongroup_10_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_Iinh)
                );
        dev_array_neurongroup_10_Iinh = 0;
    }
    if(_array_neurongroup_10_running_sum_X!=0)
    {
        delete [] _array_neurongroup_10_running_sum_X;
        _array_neurongroup_10_running_sum_X = 0;
    }
    if(dev_array_neurongroup_10_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_running_sum_X)
                );
        dev_array_neurongroup_10_running_sum_X = 0;
    }
    if(_array_neurongroup_10_v!=0)
    {
        delete [] _array_neurongroup_10_v;
        _array_neurongroup_10_v = 0;
    }
    if(dev_array_neurongroup_10_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_v)
                );
        dev_array_neurongroup_10_v = 0;
    }
    if(_array_neurongroup_10_X!=0)
    {
        delete [] _array_neurongroup_10_X;
        _array_neurongroup_10_X = 0;
    }
    if(dev_array_neurongroup_10_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_X)
                );
        dev_array_neurongroup_10_X = 0;
    }
    if(_array_neurongroup_10_Y!=0)
    {
        delete [] _array_neurongroup_10_Y;
        _array_neurongroup_10_Y = 0;
    }
    if(dev_array_neurongroup_10_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_10_Y)
                );
        dev_array_neurongroup_10_Y = 0;
    }
    if(_array_neurongroup_11_A!=0)
    {
        delete [] _array_neurongroup_11_A;
        _array_neurongroup_11_A = 0;
    }
    if(dev_array_neurongroup_11_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_A)
                );
        dev_array_neurongroup_11_A = 0;
    }
    if(_array_neurongroup_11_i!=0)
    {
        delete [] _array_neurongroup_11_i;
        _array_neurongroup_11_i = 0;
    }
    if(dev_array_neurongroup_11_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_i)
                );
        dev_array_neurongroup_11_i = 0;
    }
    if(_array_neurongroup_11_Iexc!=0)
    {
        delete [] _array_neurongroup_11_Iexc;
        _array_neurongroup_11_Iexc = 0;
    }
    if(dev_array_neurongroup_11_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_Iexc)
                );
        dev_array_neurongroup_11_Iexc = 0;
    }
    if(_array_neurongroup_11_Iexc2!=0)
    {
        delete [] _array_neurongroup_11_Iexc2;
        _array_neurongroup_11_Iexc2 = 0;
    }
    if(dev_array_neurongroup_11_Iexc2!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_Iexc2)
                );
        dev_array_neurongroup_11_Iexc2 = 0;
    }
    if(_array_neurongroup_11_indices!=0)
    {
        delete [] _array_neurongroup_11_indices;
        _array_neurongroup_11_indices = 0;
    }
    if(dev_array_neurongroup_11_indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_indices)
                );
        dev_array_neurongroup_11_indices = 0;
    }
    if(_array_neurongroup_11_v!=0)
    {
        delete [] _array_neurongroup_11_v;
        _array_neurongroup_11_v = 0;
    }
    if(dev_array_neurongroup_11_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_v)
                );
        dev_array_neurongroup_11_v = 0;
    }
    if(_array_neurongroup_11_X!=0)
    {
        delete [] _array_neurongroup_11_X;
        _array_neurongroup_11_X = 0;
    }
    if(dev_array_neurongroup_11_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_X)
                );
        dev_array_neurongroup_11_X = 0;
    }
    if(_array_neurongroup_11_Y!=0)
    {
        delete [] _array_neurongroup_11_Y;
        _array_neurongroup_11_Y = 0;
    }
    if(dev_array_neurongroup_11_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_11_Y)
                );
        dev_array_neurongroup_11_Y = 0;
    }
    if(_array_neurongroup_1_A!=0)
    {
        delete [] _array_neurongroup_1_A;
        _array_neurongroup_1_A = 0;
    }
    if(dev_array_neurongroup_1_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_A)
                );
        dev_array_neurongroup_1_A = 0;
    }
    if(_array_neurongroup_1_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_1_batch_sum_X;
        _array_neurongroup_1_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_1_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_batch_sum_X)
                );
        dev_array_neurongroup_1_batch_sum_X = 0;
    }
    if(_array_neurongroup_1_i!=0)
    {
        delete [] _array_neurongroup_1_i;
        _array_neurongroup_1_i = 0;
    }
    if(dev_array_neurongroup_1_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_i)
                );
        dev_array_neurongroup_1_i = 0;
    }
    if(_array_neurongroup_1_Iexc!=0)
    {
        delete [] _array_neurongroup_1_Iexc;
        _array_neurongroup_1_Iexc = 0;
    }
    if(dev_array_neurongroup_1_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_Iexc)
                );
        dev_array_neurongroup_1_Iexc = 0;
    }
    if(_array_neurongroup_1_Iinh!=0)
    {
        delete [] _array_neurongroup_1_Iinh;
        _array_neurongroup_1_Iinh = 0;
    }
    if(dev_array_neurongroup_1_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_Iinh)
                );
        dev_array_neurongroup_1_Iinh = 0;
    }
    if(_array_neurongroup_1_running_sum_X!=0)
    {
        delete [] _array_neurongroup_1_running_sum_X;
        _array_neurongroup_1_running_sum_X = 0;
    }
    if(dev_array_neurongroup_1_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_running_sum_X)
                );
        dev_array_neurongroup_1_running_sum_X = 0;
    }
    if(_array_neurongroup_1_v!=0)
    {
        delete [] _array_neurongroup_1_v;
        _array_neurongroup_1_v = 0;
    }
    if(dev_array_neurongroup_1_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_v)
                );
        dev_array_neurongroup_1_v = 0;
    }
    if(_array_neurongroup_1_X!=0)
    {
        delete [] _array_neurongroup_1_X;
        _array_neurongroup_1_X = 0;
    }
    if(dev_array_neurongroup_1_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_X)
                );
        dev_array_neurongroup_1_X = 0;
    }
    if(_array_neurongroup_1_Y!=0)
    {
        delete [] _array_neurongroup_1_Y;
        _array_neurongroup_1_Y = 0;
    }
    if(dev_array_neurongroup_1_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_1_Y)
                );
        dev_array_neurongroup_1_Y = 0;
    }
    if(_array_neurongroup_2_A!=0)
    {
        delete [] _array_neurongroup_2_A;
        _array_neurongroup_2_A = 0;
    }
    if(dev_array_neurongroup_2_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_A)
                );
        dev_array_neurongroup_2_A = 0;
    }
    if(_array_neurongroup_2_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_2_batch_sum_X;
        _array_neurongroup_2_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_2_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_batch_sum_X)
                );
        dev_array_neurongroup_2_batch_sum_X = 0;
    }
    if(_array_neurongroup_2_i!=0)
    {
        delete [] _array_neurongroup_2_i;
        _array_neurongroup_2_i = 0;
    }
    if(dev_array_neurongroup_2_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_i)
                );
        dev_array_neurongroup_2_i = 0;
    }
    if(_array_neurongroup_2_Iexc!=0)
    {
        delete [] _array_neurongroup_2_Iexc;
        _array_neurongroup_2_Iexc = 0;
    }
    if(dev_array_neurongroup_2_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_Iexc)
                );
        dev_array_neurongroup_2_Iexc = 0;
    }
    if(_array_neurongroup_2_Iinh!=0)
    {
        delete [] _array_neurongroup_2_Iinh;
        _array_neurongroup_2_Iinh = 0;
    }
    if(dev_array_neurongroup_2_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_Iinh)
                );
        dev_array_neurongroup_2_Iinh = 0;
    }
    if(_array_neurongroup_2_running_sum_X!=0)
    {
        delete [] _array_neurongroup_2_running_sum_X;
        _array_neurongroup_2_running_sum_X = 0;
    }
    if(dev_array_neurongroup_2_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_running_sum_X)
                );
        dev_array_neurongroup_2_running_sum_X = 0;
    }
    if(_array_neurongroup_2_v!=0)
    {
        delete [] _array_neurongroup_2_v;
        _array_neurongroup_2_v = 0;
    }
    if(dev_array_neurongroup_2_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_v)
                );
        dev_array_neurongroup_2_v = 0;
    }
    if(_array_neurongroup_2_X!=0)
    {
        delete [] _array_neurongroup_2_X;
        _array_neurongroup_2_X = 0;
    }
    if(dev_array_neurongroup_2_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_X)
                );
        dev_array_neurongroup_2_X = 0;
    }
    if(_array_neurongroup_2_Y!=0)
    {
        delete [] _array_neurongroup_2_Y;
        _array_neurongroup_2_Y = 0;
    }
    if(dev_array_neurongroup_2_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_2_Y)
                );
        dev_array_neurongroup_2_Y = 0;
    }
    if(_array_neurongroup_3_A!=0)
    {
        delete [] _array_neurongroup_3_A;
        _array_neurongroup_3_A = 0;
    }
    if(dev_array_neurongroup_3_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_A)
                );
        dev_array_neurongroup_3_A = 0;
    }
    if(_array_neurongroup_3_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_3_batch_sum_X;
        _array_neurongroup_3_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_3_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_batch_sum_X)
                );
        dev_array_neurongroup_3_batch_sum_X = 0;
    }
    if(_array_neurongroup_3_i!=0)
    {
        delete [] _array_neurongroup_3_i;
        _array_neurongroup_3_i = 0;
    }
    if(dev_array_neurongroup_3_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_i)
                );
        dev_array_neurongroup_3_i = 0;
    }
    if(_array_neurongroup_3_Iexc!=0)
    {
        delete [] _array_neurongroup_3_Iexc;
        _array_neurongroup_3_Iexc = 0;
    }
    if(dev_array_neurongroup_3_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_Iexc)
                );
        dev_array_neurongroup_3_Iexc = 0;
    }
    if(_array_neurongroup_3_Iinh!=0)
    {
        delete [] _array_neurongroup_3_Iinh;
        _array_neurongroup_3_Iinh = 0;
    }
    if(dev_array_neurongroup_3_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_Iinh)
                );
        dev_array_neurongroup_3_Iinh = 0;
    }
    if(_array_neurongroup_3_running_sum_X!=0)
    {
        delete [] _array_neurongroup_3_running_sum_X;
        _array_neurongroup_3_running_sum_X = 0;
    }
    if(dev_array_neurongroup_3_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_running_sum_X)
                );
        dev_array_neurongroup_3_running_sum_X = 0;
    }
    if(_array_neurongroup_3_v!=0)
    {
        delete [] _array_neurongroup_3_v;
        _array_neurongroup_3_v = 0;
    }
    if(dev_array_neurongroup_3_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_v)
                );
        dev_array_neurongroup_3_v = 0;
    }
    if(_array_neurongroup_3_X!=0)
    {
        delete [] _array_neurongroup_3_X;
        _array_neurongroup_3_X = 0;
    }
    if(dev_array_neurongroup_3_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_X)
                );
        dev_array_neurongroup_3_X = 0;
    }
    if(_array_neurongroup_3_Y!=0)
    {
        delete [] _array_neurongroup_3_Y;
        _array_neurongroup_3_Y = 0;
    }
    if(dev_array_neurongroup_3_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_3_Y)
                );
        dev_array_neurongroup_3_Y = 0;
    }
    if(_array_neurongroup_4_A!=0)
    {
        delete [] _array_neurongroup_4_A;
        _array_neurongroup_4_A = 0;
    }
    if(dev_array_neurongroup_4_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_A)
                );
        dev_array_neurongroup_4_A = 0;
    }
    if(_array_neurongroup_4_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_4_batch_sum_X;
        _array_neurongroup_4_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_4_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_batch_sum_X)
                );
        dev_array_neurongroup_4_batch_sum_X = 0;
    }
    if(_array_neurongroup_4_i!=0)
    {
        delete [] _array_neurongroup_4_i;
        _array_neurongroup_4_i = 0;
    }
    if(dev_array_neurongroup_4_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_i)
                );
        dev_array_neurongroup_4_i = 0;
    }
    if(_array_neurongroup_4_Iexc!=0)
    {
        delete [] _array_neurongroup_4_Iexc;
        _array_neurongroup_4_Iexc = 0;
    }
    if(dev_array_neurongroup_4_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_Iexc)
                );
        dev_array_neurongroup_4_Iexc = 0;
    }
    if(_array_neurongroup_4_Iinh!=0)
    {
        delete [] _array_neurongroup_4_Iinh;
        _array_neurongroup_4_Iinh = 0;
    }
    if(dev_array_neurongroup_4_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_Iinh)
                );
        dev_array_neurongroup_4_Iinh = 0;
    }
    if(_array_neurongroup_4_running_sum_X!=0)
    {
        delete [] _array_neurongroup_4_running_sum_X;
        _array_neurongroup_4_running_sum_X = 0;
    }
    if(dev_array_neurongroup_4_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_running_sum_X)
                );
        dev_array_neurongroup_4_running_sum_X = 0;
    }
    if(_array_neurongroup_4_v!=0)
    {
        delete [] _array_neurongroup_4_v;
        _array_neurongroup_4_v = 0;
    }
    if(dev_array_neurongroup_4_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_v)
                );
        dev_array_neurongroup_4_v = 0;
    }
    if(_array_neurongroup_4_X!=0)
    {
        delete [] _array_neurongroup_4_X;
        _array_neurongroup_4_X = 0;
    }
    if(dev_array_neurongroup_4_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_X)
                );
        dev_array_neurongroup_4_X = 0;
    }
    if(_array_neurongroup_4_Y!=0)
    {
        delete [] _array_neurongroup_4_Y;
        _array_neurongroup_4_Y = 0;
    }
    if(dev_array_neurongroup_4_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_4_Y)
                );
        dev_array_neurongroup_4_Y = 0;
    }
    if(_array_neurongroup_5_A!=0)
    {
        delete [] _array_neurongroup_5_A;
        _array_neurongroup_5_A = 0;
    }
    if(dev_array_neurongroup_5_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_A)
                );
        dev_array_neurongroup_5_A = 0;
    }
    if(_array_neurongroup_5_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_5_batch_sum_X;
        _array_neurongroup_5_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_5_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_batch_sum_X)
                );
        dev_array_neurongroup_5_batch_sum_X = 0;
    }
    if(_array_neurongroup_5_i!=0)
    {
        delete [] _array_neurongroup_5_i;
        _array_neurongroup_5_i = 0;
    }
    if(dev_array_neurongroup_5_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_i)
                );
        dev_array_neurongroup_5_i = 0;
    }
    if(_array_neurongroup_5_Iexc!=0)
    {
        delete [] _array_neurongroup_5_Iexc;
        _array_neurongroup_5_Iexc = 0;
    }
    if(dev_array_neurongroup_5_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_Iexc)
                );
        dev_array_neurongroup_5_Iexc = 0;
    }
    if(_array_neurongroup_5_Iinh!=0)
    {
        delete [] _array_neurongroup_5_Iinh;
        _array_neurongroup_5_Iinh = 0;
    }
    if(dev_array_neurongroup_5_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_Iinh)
                );
        dev_array_neurongroup_5_Iinh = 0;
    }
    if(_array_neurongroup_5_running_sum_X!=0)
    {
        delete [] _array_neurongroup_5_running_sum_X;
        _array_neurongroup_5_running_sum_X = 0;
    }
    if(dev_array_neurongroup_5_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_running_sum_X)
                );
        dev_array_neurongroup_5_running_sum_X = 0;
    }
    if(_array_neurongroup_5_v!=0)
    {
        delete [] _array_neurongroup_5_v;
        _array_neurongroup_5_v = 0;
    }
    if(dev_array_neurongroup_5_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_v)
                );
        dev_array_neurongroup_5_v = 0;
    }
    if(_array_neurongroup_5_X!=0)
    {
        delete [] _array_neurongroup_5_X;
        _array_neurongroup_5_X = 0;
    }
    if(dev_array_neurongroup_5_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_X)
                );
        dev_array_neurongroup_5_X = 0;
    }
    if(_array_neurongroup_5_Y!=0)
    {
        delete [] _array_neurongroup_5_Y;
        _array_neurongroup_5_Y = 0;
    }
    if(dev_array_neurongroup_5_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_5_Y)
                );
        dev_array_neurongroup_5_Y = 0;
    }
    if(_array_neurongroup_6_A!=0)
    {
        delete [] _array_neurongroup_6_A;
        _array_neurongroup_6_A = 0;
    }
    if(dev_array_neurongroup_6_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_A)
                );
        dev_array_neurongroup_6_A = 0;
    }
    if(_array_neurongroup_6_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_6_batch_sum_X;
        _array_neurongroup_6_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_6_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_batch_sum_X)
                );
        dev_array_neurongroup_6_batch_sum_X = 0;
    }
    if(_array_neurongroup_6_i!=0)
    {
        delete [] _array_neurongroup_6_i;
        _array_neurongroup_6_i = 0;
    }
    if(dev_array_neurongroup_6_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_i)
                );
        dev_array_neurongroup_6_i = 0;
    }
    if(_array_neurongroup_6_Iexc!=0)
    {
        delete [] _array_neurongroup_6_Iexc;
        _array_neurongroup_6_Iexc = 0;
    }
    if(dev_array_neurongroup_6_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_Iexc)
                );
        dev_array_neurongroup_6_Iexc = 0;
    }
    if(_array_neurongroup_6_Iinh!=0)
    {
        delete [] _array_neurongroup_6_Iinh;
        _array_neurongroup_6_Iinh = 0;
    }
    if(dev_array_neurongroup_6_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_Iinh)
                );
        dev_array_neurongroup_6_Iinh = 0;
    }
    if(_array_neurongroup_6_running_sum_X!=0)
    {
        delete [] _array_neurongroup_6_running_sum_X;
        _array_neurongroup_6_running_sum_X = 0;
    }
    if(dev_array_neurongroup_6_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_running_sum_X)
                );
        dev_array_neurongroup_6_running_sum_X = 0;
    }
    if(_array_neurongroup_6_v!=0)
    {
        delete [] _array_neurongroup_6_v;
        _array_neurongroup_6_v = 0;
    }
    if(dev_array_neurongroup_6_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_v)
                );
        dev_array_neurongroup_6_v = 0;
    }
    if(_array_neurongroup_6_X!=0)
    {
        delete [] _array_neurongroup_6_X;
        _array_neurongroup_6_X = 0;
    }
    if(dev_array_neurongroup_6_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_X)
                );
        dev_array_neurongroup_6_X = 0;
    }
    if(_array_neurongroup_6_Y!=0)
    {
        delete [] _array_neurongroup_6_Y;
        _array_neurongroup_6_Y = 0;
    }
    if(dev_array_neurongroup_6_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_6_Y)
                );
        dev_array_neurongroup_6_Y = 0;
    }
    if(_array_neurongroup_7_A!=0)
    {
        delete [] _array_neurongroup_7_A;
        _array_neurongroup_7_A = 0;
    }
    if(dev_array_neurongroup_7_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_A)
                );
        dev_array_neurongroup_7_A = 0;
    }
    if(_array_neurongroup_7_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_7_batch_sum_X;
        _array_neurongroup_7_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_7_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_batch_sum_X)
                );
        dev_array_neurongroup_7_batch_sum_X = 0;
    }
    if(_array_neurongroup_7_i!=0)
    {
        delete [] _array_neurongroup_7_i;
        _array_neurongroup_7_i = 0;
    }
    if(dev_array_neurongroup_7_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_i)
                );
        dev_array_neurongroup_7_i = 0;
    }
    if(_array_neurongroup_7_Iexc!=0)
    {
        delete [] _array_neurongroup_7_Iexc;
        _array_neurongroup_7_Iexc = 0;
    }
    if(dev_array_neurongroup_7_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_Iexc)
                );
        dev_array_neurongroup_7_Iexc = 0;
    }
    if(_array_neurongroup_7_Iinh!=0)
    {
        delete [] _array_neurongroup_7_Iinh;
        _array_neurongroup_7_Iinh = 0;
    }
    if(dev_array_neurongroup_7_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_Iinh)
                );
        dev_array_neurongroup_7_Iinh = 0;
    }
    if(_array_neurongroup_7_running_sum_X!=0)
    {
        delete [] _array_neurongroup_7_running_sum_X;
        _array_neurongroup_7_running_sum_X = 0;
    }
    if(dev_array_neurongroup_7_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_running_sum_X)
                );
        dev_array_neurongroup_7_running_sum_X = 0;
    }
    if(_array_neurongroup_7_v!=0)
    {
        delete [] _array_neurongroup_7_v;
        _array_neurongroup_7_v = 0;
    }
    if(dev_array_neurongroup_7_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_v)
                );
        dev_array_neurongroup_7_v = 0;
    }
    if(_array_neurongroup_7_X!=0)
    {
        delete [] _array_neurongroup_7_X;
        _array_neurongroup_7_X = 0;
    }
    if(dev_array_neurongroup_7_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_X)
                );
        dev_array_neurongroup_7_X = 0;
    }
    if(_array_neurongroup_7_Y!=0)
    {
        delete [] _array_neurongroup_7_Y;
        _array_neurongroup_7_Y = 0;
    }
    if(dev_array_neurongroup_7_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_7_Y)
                );
        dev_array_neurongroup_7_Y = 0;
    }
    if(_array_neurongroup_8_A!=0)
    {
        delete [] _array_neurongroup_8_A;
        _array_neurongroup_8_A = 0;
    }
    if(dev_array_neurongroup_8_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_A)
                );
        dev_array_neurongroup_8_A = 0;
    }
    if(_array_neurongroup_8_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_8_batch_sum_X;
        _array_neurongroup_8_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_8_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_batch_sum_X)
                );
        dev_array_neurongroup_8_batch_sum_X = 0;
    }
    if(_array_neurongroup_8_i!=0)
    {
        delete [] _array_neurongroup_8_i;
        _array_neurongroup_8_i = 0;
    }
    if(dev_array_neurongroup_8_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_i)
                );
        dev_array_neurongroup_8_i = 0;
    }
    if(_array_neurongroup_8_Iexc!=0)
    {
        delete [] _array_neurongroup_8_Iexc;
        _array_neurongroup_8_Iexc = 0;
    }
    if(dev_array_neurongroup_8_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_Iexc)
                );
        dev_array_neurongroup_8_Iexc = 0;
    }
    if(_array_neurongroup_8_Iinh!=0)
    {
        delete [] _array_neurongroup_8_Iinh;
        _array_neurongroup_8_Iinh = 0;
    }
    if(dev_array_neurongroup_8_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_Iinh)
                );
        dev_array_neurongroup_8_Iinh = 0;
    }
    if(_array_neurongroup_8_running_sum_X!=0)
    {
        delete [] _array_neurongroup_8_running_sum_X;
        _array_neurongroup_8_running_sum_X = 0;
    }
    if(dev_array_neurongroup_8_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_running_sum_X)
                );
        dev_array_neurongroup_8_running_sum_X = 0;
    }
    if(_array_neurongroup_8_v!=0)
    {
        delete [] _array_neurongroup_8_v;
        _array_neurongroup_8_v = 0;
    }
    if(dev_array_neurongroup_8_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_v)
                );
        dev_array_neurongroup_8_v = 0;
    }
    if(_array_neurongroup_8_X!=0)
    {
        delete [] _array_neurongroup_8_X;
        _array_neurongroup_8_X = 0;
    }
    if(dev_array_neurongroup_8_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_X)
                );
        dev_array_neurongroup_8_X = 0;
    }
    if(_array_neurongroup_8_Y!=0)
    {
        delete [] _array_neurongroup_8_Y;
        _array_neurongroup_8_Y = 0;
    }
    if(dev_array_neurongroup_8_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_8_Y)
                );
        dev_array_neurongroup_8_Y = 0;
    }
    if(_array_neurongroup_9_A!=0)
    {
        delete [] _array_neurongroup_9_A;
        _array_neurongroup_9_A = 0;
    }
    if(dev_array_neurongroup_9_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_A)
                );
        dev_array_neurongroup_9_A = 0;
    }
    if(_array_neurongroup_9_batch_sum_X!=0)
    {
        delete [] _array_neurongroup_9_batch_sum_X;
        _array_neurongroup_9_batch_sum_X = 0;
    }
    if(dev_array_neurongroup_9_batch_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_batch_sum_X)
                );
        dev_array_neurongroup_9_batch_sum_X = 0;
    }
    if(_array_neurongroup_9_i!=0)
    {
        delete [] _array_neurongroup_9_i;
        _array_neurongroup_9_i = 0;
    }
    if(dev_array_neurongroup_9_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_i)
                );
        dev_array_neurongroup_9_i = 0;
    }
    if(_array_neurongroup_9_Iexc!=0)
    {
        delete [] _array_neurongroup_9_Iexc;
        _array_neurongroup_9_Iexc = 0;
    }
    if(dev_array_neurongroup_9_Iexc!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_Iexc)
                );
        dev_array_neurongroup_9_Iexc = 0;
    }
    if(_array_neurongroup_9_Iinh!=0)
    {
        delete [] _array_neurongroup_9_Iinh;
        _array_neurongroup_9_Iinh = 0;
    }
    if(dev_array_neurongroup_9_Iinh!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_Iinh)
                );
        dev_array_neurongroup_9_Iinh = 0;
    }
    if(_array_neurongroup_9_running_sum_X!=0)
    {
        delete [] _array_neurongroup_9_running_sum_X;
        _array_neurongroup_9_running_sum_X = 0;
    }
    if(dev_array_neurongroup_9_running_sum_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_running_sum_X)
                );
        dev_array_neurongroup_9_running_sum_X = 0;
    }
    if(_array_neurongroup_9_v!=0)
    {
        delete [] _array_neurongroup_9_v;
        _array_neurongroup_9_v = 0;
    }
    if(dev_array_neurongroup_9_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_v)
                );
        dev_array_neurongroup_9_v = 0;
    }
    if(_array_neurongroup_9_X!=0)
    {
        delete [] _array_neurongroup_9_X;
        _array_neurongroup_9_X = 0;
    }
    if(dev_array_neurongroup_9_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_X)
                );
        dev_array_neurongroup_9_X = 0;
    }
    if(_array_neurongroup_9_Y!=0)
    {
        delete [] _array_neurongroup_9_Y;
        _array_neurongroup_9_Y = 0;
    }
    if(dev_array_neurongroup_9_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_9_Y)
                );
        dev_array_neurongroup_9_Y = 0;
    }
    if(_array_neurongroup_A!=0)
    {
        delete [] _array_neurongroup_A;
        _array_neurongroup_A = 0;
    }
    if(dev_array_neurongroup_A!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_A)
                );
        dev_array_neurongroup_A = 0;
    }
    if(_array_neurongroup_i!=0)
    {
        delete [] _array_neurongroup_i;
        _array_neurongroup_i = 0;
    }
    if(dev_array_neurongroup_i!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_i)
                );
        dev_array_neurongroup_i = 0;
    }
    if(_array_neurongroup_indices!=0)
    {
        delete [] _array_neurongroup_indices;
        _array_neurongroup_indices = 0;
    }
    if(dev_array_neurongroup_indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_indices)
                );
        dev_array_neurongroup_indices = 0;
    }
    if(_array_neurongroup_v!=0)
    {
        delete [] _array_neurongroup_v;
        _array_neurongroup_v = 0;
    }
    if(dev_array_neurongroup_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_v)
                );
        dev_array_neurongroup_v = 0;
    }
    if(_array_neurongroup_X!=0)
    {
        delete [] _array_neurongroup_X;
        _array_neurongroup_X = 0;
    }
    if(dev_array_neurongroup_X!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_X)
                );
        dev_array_neurongroup_X = 0;
    }
    if(_array_neurongroup_Y!=0)
    {
        delete [] _array_neurongroup_Y;
        _array_neurongroup_Y = 0;
    }
    if(dev_array_neurongroup_Y!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_neurongroup_Y)
                );
        dev_array_neurongroup_Y = 0;
    }
    if(_array_spikemonitor__source_idx!=0)
    {
        delete [] _array_spikemonitor__source_idx;
        _array_spikemonitor__source_idx = 0;
    }
    if(dev_array_spikemonitor__source_idx!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor__source_idx)
                );
        dev_array_spikemonitor__source_idx = 0;
    }
    if(_array_spikemonitor_count!=0)
    {
        delete [] _array_spikemonitor_count;
        _array_spikemonitor_count = 0;
    }
    if(dev_array_spikemonitor_count!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_count)
                );
        dev_array_spikemonitor_count = 0;
    }
    if(_array_spikemonitor_N!=0)
    {
        delete [] _array_spikemonitor_N;
        _array_spikemonitor_N = 0;
    }
    if(dev_array_spikemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_spikemonitor_N)
                );
        dev_array_spikemonitor_N = 0;
    }
    if(_array_statemonitor__indices!=0)
    {
        delete [] _array_statemonitor__indices;
        _array_statemonitor__indices = 0;
    }
    if(dev_array_statemonitor__indices!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor__indices)
                );
        dev_array_statemonitor__indices = 0;
    }
    if(_array_statemonitor_I!=0)
    {
        delete [] _array_statemonitor_I;
        _array_statemonitor_I = 0;
    }
    if(dev_array_statemonitor_I!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_I)
                );
        dev_array_statemonitor_I = 0;
    }
    if(_array_statemonitor_N!=0)
    {
        delete [] _array_statemonitor_N;
        _array_statemonitor_N = 0;
    }
    if(dev_array_statemonitor_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_N)
                );
        dev_array_statemonitor_N = 0;
    }
    if(_array_statemonitor_v!=0)
    {
        delete [] _array_statemonitor_v;
        _array_statemonitor_v = 0;
    }
    if(dev_array_statemonitor_v!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_statemonitor_v)
                );
        dev_array_statemonitor_v = 0;
    }
    if(_array_synapses_10_N!=0)
    {
        delete [] _array_synapses_10_N;
        _array_synapses_10_N = 0;
    }
    if(dev_array_synapses_10_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_10_N)
                );
        dev_array_synapses_10_N = 0;
    }
    if(_array_synapses_11_N!=0)
    {
        delete [] _array_synapses_11_N;
        _array_synapses_11_N = 0;
    }
    if(dev_array_synapses_11_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_11_N)
                );
        dev_array_synapses_11_N = 0;
    }
    if(_array_synapses_12_N!=0)
    {
        delete [] _array_synapses_12_N;
        _array_synapses_12_N = 0;
    }
    if(dev_array_synapses_12_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_12_N)
                );
        dev_array_synapses_12_N = 0;
    }
    if(_array_synapses_13_N!=0)
    {
        delete [] _array_synapses_13_N;
        _array_synapses_13_N = 0;
    }
    if(dev_array_synapses_13_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_13_N)
                );
        dev_array_synapses_13_N = 0;
    }
    if(_array_synapses_14_N!=0)
    {
        delete [] _array_synapses_14_N;
        _array_synapses_14_N = 0;
    }
    if(dev_array_synapses_14_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_14_N)
                );
        dev_array_synapses_14_N = 0;
    }
    if(_array_synapses_15_N!=0)
    {
        delete [] _array_synapses_15_N;
        _array_synapses_15_N = 0;
    }
    if(dev_array_synapses_15_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_15_N)
                );
        dev_array_synapses_15_N = 0;
    }
    if(_array_synapses_1_N!=0)
    {
        delete [] _array_synapses_1_N;
        _array_synapses_1_N = 0;
    }
    if(dev_array_synapses_1_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_1_N)
                );
        dev_array_synapses_1_N = 0;
    }
    if(_array_synapses_2_N!=0)
    {
        delete [] _array_synapses_2_N;
        _array_synapses_2_N = 0;
    }
    if(dev_array_synapses_2_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_2_N)
                );
        dev_array_synapses_2_N = 0;
    }
    if(_array_synapses_3_N!=0)
    {
        delete [] _array_synapses_3_N;
        _array_synapses_3_N = 0;
    }
    if(dev_array_synapses_3_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_3_N)
                );
        dev_array_synapses_3_N = 0;
    }
    if(_array_synapses_4_N!=0)
    {
        delete [] _array_synapses_4_N;
        _array_synapses_4_N = 0;
    }
    if(dev_array_synapses_4_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_4_N)
                );
        dev_array_synapses_4_N = 0;
    }
    if(_array_synapses_5_N!=0)
    {
        delete [] _array_synapses_5_N;
        _array_synapses_5_N = 0;
    }
    if(dev_array_synapses_5_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_5_N)
                );
        dev_array_synapses_5_N = 0;
    }
    if(_array_synapses_6_N!=0)
    {
        delete [] _array_synapses_6_N;
        _array_synapses_6_N = 0;
    }
    if(dev_array_synapses_6_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_6_N)
                );
        dev_array_synapses_6_N = 0;
    }
    if(_array_synapses_7_N!=0)
    {
        delete [] _array_synapses_7_N;
        _array_synapses_7_N = 0;
    }
    if(dev_array_synapses_7_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_7_N)
                );
        dev_array_synapses_7_N = 0;
    }
    if(_array_synapses_8_N!=0)
    {
        delete [] _array_synapses_8_N;
        _array_synapses_8_N = 0;
    }
    if(dev_array_synapses_8_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_8_N)
                );
        dev_array_synapses_8_N = 0;
    }
    if(_array_synapses_9_N!=0)
    {
        delete [] _array_synapses_9_N;
        _array_synapses_9_N = 0;
    }
    if(dev_array_synapses_9_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_9_N)
                );
        dev_array_synapses_9_N = 0;
    }
    if(_array_synapses_N!=0)
    {
        delete [] _array_synapses_N;
        _array_synapses_N = 0;
    }
    if(dev_array_synapses_N!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_N)
                );
        dev_array_synapses_N = 0;
    }
    if(_array_synapses_sources!=0)
    {
        delete [] _array_synapses_sources;
        _array_synapses_sources = 0;
    }
    if(dev_array_synapses_sources!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_sources)
                );
        dev_array_synapses_sources = 0;
    }
    if(_array_synapses_sources_1!=0)
    {
        delete [] _array_synapses_sources_1;
        _array_synapses_sources_1 = 0;
    }
    if(dev_array_synapses_sources_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_sources_1)
                );
        dev_array_synapses_sources_1 = 0;
    }
    if(_array_synapses_targets!=0)
    {
        delete [] _array_synapses_targets;
        _array_synapses_targets = 0;
    }
    if(dev_array_synapses_targets!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_targets)
                );
        dev_array_synapses_targets = 0;
    }
    if(_array_synapses_targets_1!=0)
    {
        delete [] _array_synapses_targets_1;
        _array_synapses_targets_1 = 0;
    }
    if(dev_array_synapses_targets_1!=0)
    {
        CUDA_SAFE_CALL(
                cudaFree(dev_array_synapses_targets_1)
                );
        dev_array_synapses_targets_1 = 0;
    }

    for(int i = 0; i < _num__array_statemonitor__indices; i++)
    {
        _dynamic_array_statemonitor_I[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_I[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_I.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_I);
    for(int i = 0; i < _num__array_statemonitor__indices; i++)
    {
        _dynamic_array_statemonitor_v[i].clear();
        thrust::device_vector<double>().swap(_dynamic_array_statemonitor_v[i]);
    }
    addresses_monitor__dynamic_array_statemonitor_v.clear();
    thrust::device_vector<double*>().swap(addresses_monitor__dynamic_array_statemonitor_v);

    // static arrays
    if(_static_array__array_synapses_sources!=0)
    {
        delete [] _static_array__array_synapses_sources;
        _static_array__array_synapses_sources = 0;
    }
    if(_static_array__array_synapses_sources_1!=0)
    {
        delete [] _static_array__array_synapses_sources_1;
        _static_array__array_synapses_sources_1 = 0;
    }
    if(_static_array__array_synapses_targets!=0)
    {
        delete [] _static_array__array_synapses_targets;
        _static_array__array_synapses_targets = 0;
    }
    if(_static_array__array_synapses_targets_1!=0)
    {
        delete [] _static_array__array_synapses_targets_1;
        _static_array__array_synapses_targets_1 = 0;
    }
    if(_timedarray_2_values!=0)
    {
        delete [] _timedarray_2_values;
        _timedarray_2_values = 0;
    }
    if(_timedarray_values!=0)
    {
        delete [] _timedarray_values;
        _timedarray_values = 0;
    }


}

