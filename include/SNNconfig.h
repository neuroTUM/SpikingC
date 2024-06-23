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

#define PATH_WEIGHTS_FC1_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc1_weights.bin"
#define PATH_BIAS_FC1_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc1_bias.bin"
#define PATH_WEIGHTS_FC2_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc2_weights.bin"
#define PATH_BIAS_FC2_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc2_bias.bin"
#define PATH_WEIGHTS_FC3_BIN    "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc3_weights.bin"
#define PATH_BIAS_FC3_BIN       "../../models/SNN_3L_simple_LIF_NMNIST/weights_and_bias_binary/fc3_bias.bin"

/* Fixed-point representation for network elements like membrane potentials, thresholds and beta values */
typedef int16_t fxp16_t;

/* Fixed-point representation for network weights and biases */
typedef int8_t fxp8_t;

/* Network parameters */
extern unsigned int layer_size[NUM_LAYERS + 1];
extern char         layer_type[NUM_LAYERS][MAX_STR_LEN];
extern char         Beta[NUM_LAYERS];                                       // "Real" beta is (2 >> Beat[i])
extern fxp16_t      threshold[NUM_LAYERS];
extern fxp16_t      L[NUM_LAYERS];

/* Statically allocated memory for weights in column major order */
extern fxp8_t W[INPUT_SIZE * L1_SIZE_OUT + 
                  LIF1_SIZE * L2_SIZE_OUT  +
                  LIF2_SIZE * L3_SIZE_OUT];

/* Statically allocated memory for biases */
extern fxp8_t B[L1_SIZE_OUT  + 
                  L2_SIZE_OUT  + 
                  L3_SIZE_OUT];

/* Statically allocated scrach pad memory used for inputs and outputs of linear layers*/
extern fxp16_t scrachpad_memory[INPUT_SIZE];

/* Statically allocated memory for membrane potentials */
extern fxp16_t mem_potential[LIF1_SIZE + LIF2_SIZE + LIF3_SIZE];

/* Linked list used to store events */
typedef struct event{
    unsigned int position;
    struct event* next;
} event_t;

extern event_t* event_list;

/* Structures for easier data handling */
typedef struct{
    fxp8_t** ptr;
    unsigned int rows;
    unsigned int cols;
} fxp8_2d_array_t;

typedef struct{
    fxp8_t* ptr;
    unsigned int size;
} fxp8_array_t;

typedef struct{
    fxp16_t* ptr;
    unsigned int size;
} fxp16_array_t;

#ifdef __cplusplus
}
#endif

#endif
