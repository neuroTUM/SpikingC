#ifndef LIF_HPP
#define LIF_HPP

#include "Utility.hpp"

/* Treba implementirati u drugoj klasi inicijalizaciju za U i inicijalizaciju za spike */

class LIF{
    public:
        LIF(unsigned int layer_num);
        ~LIF();
        void computeOutput(cfloat_array_t* In, spike_array_t* Out);
    private:

        #ifdef TEST
        static unsigned int cnt;
        unsigned int layer_idx;
        unsigned int current_time_step;
        #endif

        unsigned int layer_num;
        cfloat_array_t U;
};

#ifdef TEST
unsigned int LIF::cnt = 0;
#endif

LIF::LIF(unsigned int layer_num){

    #ifdef TEST
    cnt++;
    this->layer_idx = cnt;
    this->current_time_step = 0;
    #endif

    this->layer_num = layer_num;
    this->U.ptr = returnMemPotentialPtr(layer_num);
    this->U.size = layer_size[layer_num];
}

LIF::~LIF(){
    this->U.ptr = nullptr;
}

void LIF::computeOutput(cfloat_array_t* In, spike_array_t* Out){

    /*  
        IMPORTANT:
        Only the 'subtract' reset type is supported!

        Observations:
        1. Spikes are generated with the updated values of potentials at the same
           time step
        2. I have to save the spike values from the previous time step
        3. S[t] = (U[t] > Threshold) ? 1 : 0
           U[t] = beta * (U[t-1] - S[t-1] * Threshold) + Isyn[t]
    */

    spike_t in, out;
    unsigned char cnt;
    unsigned int idx = 0;
    for(unsigned int i = 0; i < this->U.size; i += sizeof(spike_t) * 8){
        cnt = 0;
        out = 0;
        spike_t val = Out->ptr[idx];        
        for(unsigned int j = i; j < min(i + sizeof(spike_t) * 8, this->U.size); j++){
            if(reset_type[this->layer_num]){
                in = (val >> cnt) & MASK;
                this->U.ptr[j] = Beta[this->layer_num] * (this->U.ptr[j] - in * threshold[this->layer_num]) + In->ptr[j];
                in = (this->U.ptr[j] > threshold[this->layer_num]) ? 1 : 0;
                out |= (in << cnt);
                cnt++;
            }
            else{
                /* Zero reset not supported */
            }

            #ifdef TEST
            std::string json_idx = "mem" + std::to_string(this->layer_idx);
            cfloat_t true_val = output_ref[json_idx][this->current_time_step][0][j];
            if(abs(this->U.ptr[j] - true_val) > PRECISION){
                std::cout << "Membrane potential mismatch in a LIF layer at index=" << j
                        << ", time step=" << this->current_time_step 
                        << ", layer=" << this->layer_num
                        << std::endl;
                std::cout << "Calculated value U=" << this->U.ptr[j] << ", while the true one is U_true=" << true_val << std::endl;
            }     
            #endif
        }
        Out->ptr[idx++] = out;
    }

    #ifdef TEST
    this->current_time_step++;
    #endif
}

#endif