/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: - 
*/

#include "LIF.h"

void clearLIF(lif_t* layer){
    layer->U.ptr = NULL;
}

void computeOutput(lif_t* layer, cfloat_array_t* In, spike_array_t* Out){
    
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
    for(unsigned int i = 0; i < layer->U.size; i += sizeof(spike_t) * 8){
        out = 0;
        spike_t val = Out->ptr[i / (sizeof(spike_t) * 8)];        
        for(unsigned int j = i; j < min(i + sizeof(spike_t) * 8, layer->U.size); j++){
            if(reset_type[layer->layer_num]){
                in = BITVALUE(val, j - i);
                layer->U.ptr[j] = Beta[layer->layer_num] * (layer->U.ptr[j] - in * threshold[layer->layer_num]) + In->ptr[j];
                in = (layer->U.ptr[j] > threshold[layer->layer_num]) ? 0x1 : 0x0;
                out |= (in << (j - i));
            }
            else{
                /* Zero reset not supported */
            }
        }
        Out->ptr[i / (sizeof(spike_t) * 8)] = out;
    }
}

void initLIF(lif_t* layer, unsigned int layer_num){
    layer->layer_num            = layer_num;
    layer->U.ptr                = returnMemPotentialPtr(layer_num);
    layer->U.size               = layer_size[layer_num];
    layer->clearLIF_fptr        = &clearLIF;
    layer->computeOutput_fptr   = &computeOutput;
}
