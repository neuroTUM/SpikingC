#ifndef MEMORY_ACCESS_HPP
#define MEMORY_ACCESS_HPP

#include "network_config.hpp"

template<typename T>
T** returnWeightPtr(unsigned int layer_num){
    unsigned int offset = 0;
    T** ptr;
    ptr = new T*[layer_size[layer_num + 1]];

    for(unsigned int i = 0; i < layer_num; i++){
        offset += layer_size[i] * layer_size[i + 1];
    }

    for(unsigned int i = 0; i < layer_size[layer_num + 1]; i++){
        ptr[i] = &W[i * layer_size[layer_num] + offset];
    }
    return ptr;
}

template<typename T>
T* returnBiasPtr(unsigned int layer_num){
    unsigned long long idx = 0;
    for(unsigned int i = 0; i < layer_num; i++){
        idx += layer_size[i + 1];
    }
    return &B[idx];
}

template<typename T>
T** returnInputPtr(unsigned int input_num){
    T** ptr;
    ptr = new T*[TIME_STEPS];    
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        ptr[i] = &input[(i * INPUT_SIZE) + (input_num * INPUT_SIZE * TIME_STEPS)];
    }
    return ptr;
}

#endif