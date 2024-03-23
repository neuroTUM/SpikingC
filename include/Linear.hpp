#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "Utility.hpp"

class Linear{
    public:
        Linear(unsigned int layer_num);
        ~Linear();
        void computeOutput(cfloat_array_t* In, cfloat_array_t* Out);
        void computeOutput(spike_array_t* In, cfloat_array_t* Out);
    private:

        #ifdef TEST
        static unsigned int cnt;
        unsigned int layer_idx;
        unsigned int layer_num;
        unsigned int current_time_step;
        #endif

        wfloat_2d_array_t W;
        wfloat_array_t B;
};

#ifdef TEST
unsigned int Linear::cnt = 0;
#endif

Linear::Linear(unsigned int layer_num){
    #ifdef TEST
    cnt++;
    this->layer_idx = cnt;
    this->layer_num = layer_num;
    this->current_time_step = 0;
    #endif

    this->W.ptr     = returnWeightPtr(layer_num);
    this->W.rows    = layer_size[layer_num + 1];
    this->W.cols    = layer_size[layer_num];
    this->B.ptr     = returnBiasPtr(layer_num);
    this->B.size    = layer_size[layer_num + 1];
}

Linear::~Linear(){
    delete[] this->W.ptr;
    this->W.ptr = nullptr;
    this->B.ptr = nullptr;
}

void Linear::computeOutput(cfloat_array_t* In, cfloat_array_t* Out){

    matrixVectorMul(&(this->W), &(this->B), In, Out);

    #ifdef TEST
    for(unsigned int i = 0; i < Out->size; i++){
        std::string json_idx = "fc" + std::to_string(this->layer_idx) + "_outputs";
        cfloat_t true_val = output_ref[json_idx][this->current_time_step][0][i];
        if(abs(Out->ptr[i] - true_val) > PRECISION){
            std::cout << "Output value mismatch in a fully connected layer at index=" << i 
                        << ", time step=" << this->current_time_step 
                        << ", layer=" << this->layer_num
                        << std::endl;
            std::cout << "Calculated value Isyn=" << Out->ptr[i] << ", while the true one is fc_true=" << true_val << std::endl;
        }
    }     
    this->current_time_step++;
    #endif
}

void Linear::computeOutput(spike_array_t* In, cfloat_array_t* Out){
    matrixVectorMulSparse(&(this->W), &(this->B), In, Out);

    #ifdef TEST
    for(unsigned int i = 0; i < Out->size; i++){
        std::string json_idx = "fc" + std::to_string(this->layer_idx) + "_outputs";
        cfloat_t true_val = output_ref[json_idx][this->current_time_step][0][i];
        if(abs(Out->ptr[i] - true_val) > PRECISION){
            std::cout << "Output value mismatch in a fully connected layer at index=" << i 
                        << ", time step=" << this->current_time_step 
                        << ", layer=" << this->layer_num
                        << std::endl;
            std::cout << "Calculated value Isyn=" << Out->ptr[i] << ", while the true one is fc_true=" << true_val << std::endl;
        }
    }     
    this->current_time_step++;
    #endif    
}

#endif