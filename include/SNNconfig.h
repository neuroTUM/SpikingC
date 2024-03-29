#ifndef SNNCONFIGCONFIG_H
#define SNNCONFIG_H

/* Common includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test related macros */
#define TEST
#define LOAD
#define PRECISION 1e-4
//#define DATALOADER

#ifdef DATALOADER
#include <dirent.h> // For directory operations
#include <stdint.h> // For int16_t type
#include <sys/types.h>
#endif

/* General macros */
#define MAX_STR_LEN 10
#define TRUE 1
#define FALSE 0

/* Macros used for defining layer parameters and inputs */
#define TIME_STEPS 31
#define NUM_LAYERS 6
#define INPUT_SIZE 2312
#define L1_SIZE_OUT 2312 / 4
#define LIF1_SIZE 2312 / 4
#define L2_SIZE_OUT 2312 / 8
#define LIF2_SIZE 2312 / 8
#define L3_SIZE_OUT 10
#define LIF3_SIZE 10

#define PATH_WEIGHTS_FC1 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc1_weight_weights.csv"
#define PATH_BIAS_FC1 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc1_bias.csv"
#define PATH_WEIGHTS_FC2 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc2_weight_weights.csv"
#define PATH_BIAS_FC2 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc2_bias.csv"
#define PATH_WEIGHTS_FC3 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc3_weight_weights.csv"
#define PATH_BIAS_FC3 "../models/SNN_3L_simple_LIF_NMNIST/weights_and_biases/fc3_bias.csv"

/* Floating point representation for network elements like membrane potentials, thresholds and beta values */
typedef double cfloat_t;

/* Floating point representation for network parameters */
typedef double wfloat_t;

/* Data type used for representing spike data */
typedef unsigned char spike_t;

/* Data type simulating bool in C */
typedef unsigned char bool_t;

/* Network parameters */
unsigned int layer_size[NUM_LAYERS + 1] = {INPUT_SIZE, L1_SIZE_OUT, LIF1_SIZE, L2_SIZE_OUT, LIF2_SIZE, L3_SIZE_OUT, LIF3_SIZE};
char layer_type[NUM_LAYERS][MAX_STR_LEN] = {"Linear", "LIF", "Linear", "LIF", "Linear", "LIF"};
cfloat_t Beta[NUM_LAYERS] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
cfloat_t threshold[NUM_LAYERS] = {0, 2.5, 0, 8.0, 0, 4.0};
bool_t reset_type[NUM_LAYERS] = {TRUE, TRUE, TRUE, TRUE, TRUE, TRUE};

/* Statically allocated memory for weights in row major order */
wfloat_t W[INPUT_SIZE * L1_SIZE_OUT + 
           LIF1_SIZE * L2_SIZE_OUT    +
           LIF2_SIZE * L3_SIZE_OUT];

/* Statically allocated memory for biases */
wfloat_t B[L1_SIZE_OUT  + 
           L2_SIZE_OUT  + 
           L3_SIZE_OUT];

/* Statically allocated scrach pad memory used for inputs and outputs of linear layers*/
cfloat_t scrachpad_memory[INPUT_SIZE + L1_SIZE_OUT];

/* Statically allocated scrach pad memory used outputs of LIF layers*/
spike_t spike_memory[((LIF1_SIZE + LIF2_SIZE + LIF3_SIZE) / 8) + (NUM_LAYERS / 2)];

/* Statically allocated memory for membrane potentials */
cfloat_t mem_potential[LIF1_SIZE + LIF2_SIZE + LIF3_SIZE];

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
#endif

#endif