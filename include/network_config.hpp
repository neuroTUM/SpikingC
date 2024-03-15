#ifndef NETWORK_CONFIG_HPP
#define NETWORK_CONFIG_HPP

// #define TEST
#define LOAD
#define PRECISION 1e-4

#include <iostream>

/* Macros used for defining layer parameters and inputs */
#define NUM_LAYERS 3
#define L1_SIZE 2312 / 4
#define L2_SIZE 2312 / 8
#define L3_SIZE 10
#define INPUT_SIZE 2312
#define TIME_STEPS 31
#define NUM_OF_INPUT_IMAGES 1


/* Different data types used for weights, biases and further parameters */
typedef double fp_t;
typedef double weight_t;

unsigned int layer_size[NUM_LAYERS + 1] = {INPUT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE};
fp_t beta[NUM_LAYERS] = {0.5, 0.5, 0.5};
fp_t threshold[NUM_LAYERS] = {2.5, 8.0, 4.0};
bool reset_type[NUM_LAYERS] = {true, true, true};

/* Static arrays for weights, biases and inputs */

/* Each weight matrix is stored in memory in row major fashion */
weight_t W[INPUT_SIZE * L1_SIZE + 
           L1_SIZE * L2_SIZE    +
           L2_SIZE * L3_SIZE];

weight_t B[L1_SIZE  + 
           L2_SIZE  + 
           L3_SIZE];

fp_t input[INPUT_SIZE * TIME_STEPS * NUM_OF_INPUT_IMAGES];

#ifdef LOAD
#include <string>
#include "../include/json.hpp"
#include <fstream>

using json = nlohmann::json;
std::string path_to_weights = "../../models/SNN_3L_simple_LIF_NMNIST/extra_formats/model_weights.json";
std::string path_to_inputs = "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json";

void loadWeights(){

    unsigned int idx = 0;
    std::ifstream f(path_to_weights);
    json data = json::parse(f);

    for(unsigned int i = 0; i < INPUT_SIZE * L1_SIZE; i++){
        W[idx++] = data["fc1.weight"][i];
    }

    for(unsigned int i = 0; i < L1_SIZE * L2_SIZE; i++){
        W[idx++] = data["fc2.weight"][i];
    }

    for(unsigned int i = 0; i < L2_SIZE * L3_SIZE; i++){
        W[idx++] = data["fc3.weight"][i];
    }

    idx = 0;
    for(unsigned int i = 0; i < L1_SIZE; i++){
        B[idx++] = data["fc1.bias"][i];
    }

    for(unsigned int i = 0; i < L2_SIZE; i++){
        B[idx++] = data["fc2.bias"][i];
    }

    for(unsigned int i = 0; i < L3_SIZE; i++){
        B[idx++] = data["fc3.bias"][i];
    }
}

void loadInputs(){
    unsigned int idx = 0;
    std::ifstream f(path_to_inputs);
    json data = json::parse(f);

    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < INPUT_SIZE; j++){
            input[idx++] = data["inputs"][i][0][j];
        }
    }
}

#endif

#endif