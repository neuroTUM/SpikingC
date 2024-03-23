#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <algorithm>
#include "SNNconfig.hpp"

unsigned int min(unsigned int x, unsigned int y){
    if(x > y)
        return y;
    return x;
}

wfloat_t** returnWeightPtr(unsigned int layer_num){
    unsigned int offset = 0;
    wfloat_t** ptr;
    ptr = new wfloat_t*[layer_size[layer_num + 1]];

    for(unsigned int i = 0; i < layer_num; i++){
        if(layer_type[i] == "Linear")
            offset += layer_size[i] * layer_size[i + 1];
    }

    for(unsigned int i = 0; i < layer_size[layer_num + 1]; i++){
        ptr[i] = &W[i * layer_size[layer_num] + offset];
    }
    return ptr;
}

wfloat_t* returnBiasPtr(unsigned int layer_num){
    unsigned long long idx = 0;
    for(unsigned int i = 0; i < layer_num; i++){
        if(layer_type[i] == "Linear")
            idx += layer_size[i + 1];
    }
    return &B[idx];
}

cfloat_t* returnMemPotentialPtr(unsigned int layer_num){
    unsigned long long idx = 0;
    for(unsigned int i = 0; i < layer_num; i++){
        if(layer_type[i] == "LIF")
            idx += layer_size[i + 1];
    }
    return &mem_potential[idx];
}

spike_t* returnSpikePtr(unsigned int layer_num){
    unsigned long long idx = 0;
    for(unsigned int i = 0; i < layer_num; i++){
        if(layer_type[i] == "LIF")
            idx += (layer_size[i + 1] / (sizeof(spike_t) * 8)) + 1;
    }
    return &spike_memory[idx];
}

void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, cfloat_array_t* In, cfloat_array_t* Out){

    if((W->cols != In->size) || (W->rows != B->size) || (Out->size != B->size)){
        std::cout << "matrixVectorMul : Inappropriate dimensions" << std::endl; 
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
        std::cout << "matrixVectorMulSparse : Inappropriate dimensions" << std::endl; 
        exit(1);        
    }

    cfloat_t r;
    spike_t val;
    unsigned char cnt;
    unsigned int idx;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        idx = 0;
        for(unsigned int j = 0; j < W->cols; j += sizeof(spike_t) * 8){
            val = In->ptr[idx++];
            if(val == 0)
                continue;
            else{
                cnt = 0;
                for(unsigned int k = j; k < min(j + sizeof(spike_t) * 8, W->cols); k++){
                    if((val >> cnt) & MASK)
                        r += W->ptr[i][k];
                    cnt++;
                }
            } 
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}

int extractLabelFromFilename(const std::string &filename)
{
    size_t pos = filename.rfind('_');
    if (pos != std::string::npos)
    {
        size_t start = pos + 1;
        size_t end = filename.rfind('.');
        std::string labelStr = filename.substr(start, end - start);
        return std::stoi(labelStr);
    }
    return -1;
}

#endif