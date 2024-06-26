#ifndef SNNCONFIGCONFIG_H
#define SNNCONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Common includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <stdint.h>

/* Test related macros */
//#define TEST
//#define BINARY_IMPLEMENTATION
#define DATALOADER
//#define PRINT_WnB


#ifdef DATALOADER
#include <dirent.h> // For directory operations
#include <sys/types.h>
#endif

#ifdef TEST
#include <math.h>
#define PRECISION 1e-4
#endif

/* General macros */
#define MAX_STR_LEN 10

/* Macros used for defining layer parameters and inputs */
#define TIME_STEPS  31
#define CHANNELS     2
#define HEIGHT      34
#define WIDTH       34

#define DATA_POINTS_PER_TIMESTEP (CHANNELS * HEIGHT * WIDTH) // Total data points per timestep
#define INPUT_STEP_SIZE (CHANNELS * HEIGHT * WIDTH * sizeof(int16_t))

#define NUM_LAYERS  6
#define INPUT_SIZE  2312
#define L1_SIZE_OUT 2312 / 4
#define LIF1_SIZE   2312 / 4
#define L2_SIZE_OUT 2312 / 8
#define LIF2_SIZE   2312 / 8
#define L3_SIZE_OUT 10
#define LIF3_SIZE   10

#ifdef BINARY_IMPLEMENTATION
#define PATH_WEIGHTS_FC1_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc1_weight.bin"
#define PATH_BIAS_FC1_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc1_bias.bin"
#define PATH_WEIGHTS_FC2_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc2_weight.bin"
#define PATH_BIAS_FC2_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc2_bias.bin"
#define PATH_WEIGHTS_FC3_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc3_weight.bin"
#define PATH_BIAS_FC3_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc3_bias.bin"
#else
#define PATH_WEIGHTS_FC1        "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc1_weights.csv"
#define PATH_BIAS_FC1           "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc1_bias.csv"
#define PATH_WEIGHTS_FC2        "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc2_weights.csv"
#define PATH_BIAS_FC2           "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc2_bias.csv"
#define PATH_WEIGHTS_FC3        "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc3_weights.csv"
#define PATH_BIAS_FC3           "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias/fc3_bias.csv"
#endif

/* Floating point representation for network elements like membrane potentials, thresholds and beta values */
typedef float cfloat_t;

/* Floating point representation for network parameters */
typedef float wfloat_t;

/* Data type used for representing spike data */
typedef unsigned char spike_t;

/* Network parameters */
extern unsigned int layer_size[NUM_LAYERS + 1];
extern char         layer_type[NUM_LAYERS][MAX_STR_LEN];
extern cfloat_t     Beta[NUM_LAYERS];
extern cfloat_t     threshold[NUM_LAYERS];

/* Statically allocated memory for weights in row major order */
extern wfloat_t W[INPUT_SIZE * L1_SIZE_OUT + 
                  LIF1_SIZE * L2_SIZE_OUT    +
                  LIF2_SIZE * L3_SIZE_OUT];

/* Statically allocated memory for biases */
extern wfloat_t B[L1_SIZE_OUT  + 
                  L2_SIZE_OUT  + 
                  L3_SIZE_OUT];

/* Statically allocated scrach pad memory used for inputs and outputs of linear layers*/
extern cfloat_t scrachpad_memory[INPUT_SIZE + L1_SIZE_OUT];

/* Statically allocated scrach pad memory used outputs of LIF layers*/
extern spike_t spike_memory[((LIF1_SIZE + LIF2_SIZE + LIF3_SIZE) / 8) + (NUM_LAYERS / 2)];

/* Statically allocated memory for membrane potentials */
extern cfloat_t mem_potential[LIF1_SIZE + LIF2_SIZE + LIF3_SIZE];

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

#ifdef __cplusplus
}
#endif

#endif
