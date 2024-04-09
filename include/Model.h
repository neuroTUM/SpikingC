/*
    Author's name: Aleksa Stojkovic
    Date of creation: 28.3.2024
    Description: A set of structures and functions used to create models containing linear and LIF layers
*/

#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "LIF.h"
#include "Linear.h"

/* COMMENTS:
   - In embedded systems dynamic memory allocation should be avoided in order to avoid memory fragmentations
     that could lead to malloc failure. Embedded devices are often not accessible and therefore no repairs/fixes
     are possible.
   - Model.h and Model.c are just overheads, because in reality an embedded system just runs the same code and no
     changes are possible, hence a better option is just to hardcode one version of a network. On the other hand,
     I find this set of functions and structures useful for design space exploration until the right network configuration
     is found and it makes it easier to use, especially for people new to the project.
   - The decision to use function pointers was made just to make it clearer for the newer users what functions are tied to
     what structures. This design decision is suceptible to change in future versions. 
   - Enrico's suggestion to use tagged unions was accepted as it results in better organized and cleaner code in my opinion.  
*/

/* Union containing pointers to lif and linear layers */
typedef union{
    lif_t* lif_ptr;
    linear_t* linear_ptr;
} layer_instance_t;

typedef struct Model{
    /* A pointer to an array of layers */
    layer_instance_t* layers;
    cfloat_array_t floatOut;
    spike_array_t spikeOut;
    unsigned int *actPred;
    void (*clearModel_fptr) (struct Model*);
    void (*resetState_fptr) (struct Model*);
    void (*run_fptr) (struct Model*, cfloat_array_t*);
    unsigned int (*predict_fptr) (struct Model*);
} model_t;

/* Allocates memory for each layer and initializes the necessary structures */
void initModel(model_t*);

/* Deallocates previously allocated memory */
void clearModel(model_t*);

/* Resets all the membrane potentials, spikes and active predictions */
void resetState(model_t*);

/* Implements the forward path for one time step */
void run(model_t*, cfloat_array_t*);

/* Predicts the output based on the active prediction arrays */
unsigned int predict(model_t*);

#ifdef __cplusplus
}
#endif

#endif