/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: -
*/

#include "../include/Utility.h"

unsigned int min(unsigned int x, unsigned int y){
    if(x > y)
        return y;
    return x;
}

unsigned int getOffset(unsigned int layer_num, char offset_type, char* str){
    unsigned int offset = 0;

    for(unsigned int i = 0; i < layer_num; i++){
        if(strcmp(layer_type[i], "Linear") == 0){
            if(offset_type == 'M')
                offset += layer_size[i] * layer_size[i + 1];
            else if(offset_type == 'V')
                offset += layer_size[i + 1];
        }
        else if(strcmp(layer_type[i], "LIF") == 0){
            if(offset_type == 'V')
                offset += layer_size[i + 1];
            else if(offset_type == 'S')
                offset += (layer_size[i + 1] / (sizeof(spike_t) * 8)) + 1;            
        }
    }

    return offset;
}

wfloat_t** returnWeightPtr(unsigned int layer_num){
    wfloat_t** ptr;
    ptr = malloc(sizeof(wfloat_t*) * layer_size[layer_num + 1]);

    for(unsigned int i = 0; i < layer_size[layer_num + 1]; i++){
        ptr[i] = &W[i * layer_size[layer_num] + getOffset(layer_num, 'M', "Linear")];
    }
    return ptr;
}

wfloat_t* returnBiasPtr(unsigned int layer_num){
    return &B[getOffset(layer_num, 'V', "Linear")];
}

cfloat_t* returnMemPotentialPtr(unsigned int layer_num){
    return &mem_potential[getOffset(layer_num, 'V', "LIF")];
}

spike_t* returnSpikePtr(unsigned int layer_num){
    return &spike_memory[getOffset(layer_num, 'S', "LIF")];
}

void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, cfloat_array_t* In, cfloat_array_t* Out){

    if((W->cols != In->size) || (W->rows != B->size) || (Out->size != B->size)){
        printf("matrixVectorMul : Inappropriate dimensions\n"); 
        exit(1);        
    }

    cfloat_t r;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        for(unsigned int j = 0; j < W->cols; j++){
            r += W->ptr[i][j] * In->ptr[j];
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}

void matrixVectorMulSparse(wfloat_2d_array_t* W, wfloat_array_t* B, spike_array_t* In, cfloat_array_t* Out){

    if((W->cols != In->size) || (W->rows != B->size) || (Out->size != B->size)){
        printf("matrixVectorMulSparse : Inappropriate dimensions\n"); 
        exit(1);        
    }

    cfloat_t r;
    spike_t val;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        for(unsigned int j = 0; j < W->cols; j += sizeof(spike_t) * 8){
            val = In->ptr[j / (sizeof(spike_t) * 8)];
            if(val == 0)
                continue;
            else{
                for(unsigned int k = j; k < min(j + sizeof(spike_t) * 8, W->cols); k++){
                    if(BITVALUE(val, k - j))
                        r += W->ptr[i][k];
                }
            } 
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}