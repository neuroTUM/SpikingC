#include <iostream>
#include <string>
#include <chrono>
#include "../include/fully_connected.hpp"
#include "../include/leaky.hpp"

using namespace std;

int main()
{   
    #ifdef LOAD
    loadWeights();
    loadInputs();
    #endif

    fp_t** inout = returnInputPtr<fp_t>(0);

    fully_connected<double> FC1(0);
    leaky<double> LEAKY1(0);
    fully_connected<double> FC2(1);
    leaky<double> LEAKY2(1);
    fully_connected<double> FC3(2);
    leaky<double> LEAKY3(2);

    inout = FC1.run(inout);
    inout = LEAKY1.run(inout);
    
    #ifdef TEST
    std::ifstream f(path_to_inputs);
    json data = json::parse(f);
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < layer_size[1]; j++){
            std::string json_idx = "lif" + std::to_string(1) + "_spikes";
            fp_t true_val = data[json_idx][i][0][j];
            if(abs(inout[i][j] - true_val) > PRECISION){
                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
                          << ", time step=" << i 
                          << ", layer=" << 0
                          << std::endl;
                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
            }
        }
    }          
    #endif    

    inout = FC2.run(inout);
    inout = LEAKY2.run(inout);

    #ifdef TEST
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < layer_size[2]; j++){
            std::string json_idx = "lif" + std::to_string(2) + "_spikes";
            fp_t true_val = data[json_idx][i][0][j];
            if(abs(inout[i][j] - true_val) > PRECISION){
                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
                          << ", time step=" << i 
                          << ", layer=" << 1
                          << std::endl;
                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
            }
        }
    }          
    #endif

    inout = FC3.run(inout);
    inout = LEAKY3.run(inout);

    #ifdef TEST
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < layer_size[3]; j++){
            std::string json_idx = "lif" + std::to_string(3) + "_spikes";
            fp_t true_val = data[json_idx][i][0][j];
            if(abs(inout[i][j] - true_val) > PRECISION){
                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
                          << ", time step=" << i 
                          << ", layer=" << 2
                          << std::endl;
                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
            }
        }
    }          
    #endif
    
    cout << "Predicted class is: " << LEAKY3.predictClass() << endl;

    for(unsigned int i = 0; i < TIME_STEPS; i++){
        delete[] inout[i];
    }
    delete[] inout;
    inout = nullptr;

    return 0;
}