/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: A set of structures and functions used to describe and perform computations for a LIF layer 
*/

#ifndef LIF_H
#define LIF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Utility.h"

/* A structure holding all relevant data for LIF layers */
typedef struct LIF{
    unsigned int layer_num;
    cfloat_array_t U;
    /* A pointer to the clearLIF function */
    void (*clearLIF_fptr) (struct LIF*);
    /* A pointer to the computeOutput function */
    void (*computeOutput_fptr) (struct LIF*, cfloat_array_t*, spike_array_t*);

    #ifdef TEST
    /* Current time step */
    unsigned int curr_time_step;
    #endif
} lif_t;

/* Initializes a pointer to the membrane potential for this layer */
void initLIF(lif_t* layer, unsigned int layer_num);

/* Sets the pointer to the membrane potential to NULL */
void clearLIF(lif_t* layer);

/* Updates the membrane potentials and generates spikes if it is necessary */
void computeOutput(lif_t* layer, cfloat_array_t* In, spike_array_t* Out);

#ifdef TEST
/* Tests if membrane potentials and spikes had been calculated correctly */
void testLIF(lif_t* layer, const spike_array_t* spikes);
#endif

#ifdef __cplusplus
}
#endif

#endif