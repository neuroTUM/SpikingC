#ifndef LIF_H
#define LIF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Utility.h"

/* A structure holding all relevant data for LIF layers */
typedef struct LIF{
    unsigned int layer_num;
    fxp16_array_t U;
    /* A pointer to the clearLIF function */
    void (*clearLIF_fptr) (struct LIF*);
    /* A pointer to the computeOutput function */
    void (*computeOutput_fptr) (struct LIF*, fxp16_array_t*);
} lif_t;

/**
 * Initializes all fields in a structure used to represent a LIF layer.
 * Pointer for membrane potentials will have the right value after the execution of this function.
 * @param layer A pointer to a structure representing a LIF layer.
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Nothing is returned.
 */
void initLIF(lif_t* layer, unsigned int layer_num);

/**
 * Sets the pointer to the membrane potential to NULL.
 * @param layer A pointer to a structure representing a LIF layer.
 * @return Nothing is returned.
 */
void clearLIF(lif_t* layer);

/**
 * Updates the membrane potentials and generates spikes if it is necessary.
 * @param layer A pointer to a structure representing a LIF layer.
 * @param In A structure containing the pointer to the first element of the input vector consisting of floats used for this layer and its dimensions.
 * @return Nothing is returned.
 */
void computeOutput(lif_t* layer, fxp16_array_t* In);

#ifdef __cplusplus
}
#endif

#endif