#ifndef SNNCONFIGCONFIG_HPP
#define SNNCONFIG_HPP

// #define TEST
#define LOAD
#define PRECISION 1e-4

#include <iostream>
#include <string>

/* Macros used for defining layer parameters and inputs */
#define NUM_LAYERS 6
#define INPUT_SIZE 2312
#define L1_SIZE 2312 / 4
#define L2_SIZE 2312 / 4
#define L3_SIZE 2312 / 8
#define L4_SIZE 2312 / 8
#define L5_SIZE 10
#define L6_SIZE 10

#define MASK 0x01

/* Floating point representation for network elements like membrane potentials, thresholds and beta values */
typedef double cfloat_t;

/* Floating point representation for network parameters */
typedef double wfloat_t;

/* Data type used for representing spike data */
typedef unsigned char spike_t;

/* Network parameters */
unsigned int layer_size[NUM_LAYERS + 1] = {INPUT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE, L4_SIZE, L5_SIZE, L6_SIZE};
std::string layer_type[NUM_LAYERS] = {"Linear", "LIF", "Linear", "LIF", "Linear", "LIF"};
cfloat_t Beta[NUM_LAYERS] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
cfloat_t threshold[NUM_LAYERS] = {0, 2.5, 0, 8.0, 0, 4.0};
bool reset_type[NUM_LAYERS] = {true, true, true, true, true, true};

/* Statically allocated memory for weights in row major order */
wfloat_t W[INPUT_SIZE * L1_SIZE + 
           L2_SIZE * L3_SIZE    +
           L4_SIZE * L5_SIZE];

/* Statically allocated memory for biases */
wfloat_t B[L1_SIZE  + 
           L3_SIZE  + 
           L5_SIZE];

/* Statically allocated scrach pad memory used for inputs and outputs of linear layers*/
cfloat_t scrachpad_memory[INPUT_SIZE + L1_SIZE];

/* Statically allocated scrach pad memory used outputs of LIF layers*/
spike_t spike_memory[((L2_SIZE + L4_SIZE + L6_SIZE) / 8) + (NUM_LAYERS / 2)];

/* Statically allocated memory for membrane potentials */
cfloat_t mem_potential[L2_SIZE + L4_SIZE + L6_SIZE];

/* Structures for easier data handling */
typedef struct{
    wfloat_t** ptr;
    unsigned int rows;
    unsigned int cols;
} wfloat_2d_array_t;

typedef struct{
    wfloat_t* ptr;
    unsigned int size;
} wfloat_array_t;

typedef struct{
    cfloat_t* ptr;
    unsigned int size;
} cfloat_array_t;

typedef struct{
    spike_t* ptr;
    unsigned int size;
} spike_array_t;

#ifdef LOAD
#include "../include/json.hpp"
#include <fstream>

using json = nlohmann::json;
std::string path_to_weights = "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/model_weights.json";
std::string path_to_inputs = "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json";
json output_ref;

void parseReferenceJSON(){
    std::ifstream f(path_to_inputs);
    output_ref = json::parse(f);    
}

void loadWeights(){

    unsigned int idx = 0;
    std::ifstream f(path_to_weights);
    json data = json::parse(f);

    for(unsigned int i = 0; i < INPUT_SIZE * L1_SIZE; i++){
        W[idx++] = data["fc1.weight"][i];
    }

    for(unsigned int i = 0; i < L2_SIZE * L3_SIZE; i++){
        W[idx++] = data["fc2.weight"][i];
    }

    for(unsigned int i = 0; i < L4_SIZE * L5_SIZE; i++){
        W[idx++] = data["fc3.weight"][i];
    }

    idx = 0;
    for(unsigned int i = 0; i < L1_SIZE; i++){
        B[idx++] = data["fc1.bias"][i];
    }

    for(unsigned int i = 0; i < L3_SIZE; i++){
        B[idx++] = data["fc2.bias"][i];
    }

    for(unsigned int i = 0; i < L5_SIZE; i++){
        B[idx++] = data["fc3.bias"][i];
    }
}

#endif

#endif