#include "../include/LIF.h"

void clearLIF(lif_t* layer){
    layer->U.ptr = NULL;
}

void computeOutput(lif_t* layer, cfloat_array_t* In){

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
        layer->U.ptr[i] = Beta[layer->layer_num] * layer->U.ptr[i];
        layer->U.ptr[i] -= spike * Beta[layer->layer_num] * threshold[layer->layer_num];
        layer->U.ptr[i] += In->ptr[i];
    }
}

void initLIF(lif_t* layer, unsigned int layer_num){
    layer->layer_num            = layer_num;
    layer->U.ptr                = returnMemPotentialPtr(layer_num);
    layer->U.size               = layer_size[layer_num];
    layer->clearLIF_fptr        = &clearLIF;
    layer->computeOutput_fptr   = &computeOutput;
}