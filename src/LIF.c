#include "../include/LIF.h"

void clearLIF(lif_t* layer){
    layer->U.ptr = NULL;
}

void computeOutput(lif_t* layer, fxp16_array_t* In){

    bool spike;
    emptyList();
    for(unsigned int i = 0; i < layer->U.size; i++){
        if(layer->U.ptr[i] > threshold[layer->layer_num]){
            spike = 1;
            pushToList(i);
        }
        else{
            spike = 0;
        }

        /* IMPORTANT: All checks for over/underflow are a part of the custom instruction and shouldn't be
                      visible in the C code. Here I only want to simulate the effect those instructions will 
                      have once they are implemented.
        */

        /* When performing the shifht right operation we don't have to worry about over/underflow */
        layer->U.ptr[i] = (layer->U.ptr[i] >> Beta[layer->layer_num]);
        
        /* Check for underflow */
        if(((int)layer->U.ptr[i] - (int)(spike * L[layer->layer_num])) < INT16_MIN){
            layer->U.ptr[i] = INT16_MIN;
        }
        else{
            layer->U.ptr[i] -= spike * L[layer->layer_num];
        }

        /* Saturate the synaptic current if it goes outside of the predefined range */
        int temp = In->ptr[i] * 2;
        if(temp < INT16_MIN){
            In->ptr[i] = INT16_MIN;
        }
        else if(temp > INT16_MAX){
            In->ptr[i] = INT16_MAX;
        }
        else{
            In->ptr[i] = In->ptr[i] * 2;
        }

        /* Saturate the membrane potential value if it goes outside of the predefined range */
        if((int)layer->U.ptr[i] + (int)In->ptr[i] < INT16_MIN){
            layer->U.ptr[i] = INT16_MIN;
        }
        else if((int)layer->U.ptr[i] + (int)In->ptr[i] > INT16_MAX){
            layer->U.ptr[i] = INT16_MAX;
        }
        else{
            layer->U.ptr[i] += In->ptr[i];
        }
    }
}

void initLIF(lif_t* layer, unsigned int layer_num){
    layer->layer_num            = layer_num;
    layer->U.ptr                = returnMemPotentialPtr(layer_num);
    layer->U.size               = layer_size[layer_num];
    layer->clearLIF_fptr        = &clearLIF;
    layer->computeOutput_fptr   = &computeOutput;
}