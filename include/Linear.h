/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: A set of structures and functions used to describe and perform computations for a linear layer 
*/

#ifndef LINEAR_H
#define LINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "Utility.h"

/* A structure holding all relevant data for linear layers */
typedef struct Linear{
    unsigned int layer_num;
    wfloat_2d_array_t W;
    wfloat_array_t B;
    /* A pointer to the clearLinear function */
    void (*clearLinear_fptr) (struct Linear*);
    void (*matrixVectorMul_fptr) (wfloat_2d_array_t*, wfloat_array_t*, cfloat_array_t*, cfloat_array_t*);
    void (*matrixVectorMulSparse_fptr) (wfloat_2d_array_t*, wfloat_array_t*, spike_array_t*, cfloat_array_t*);
} linear_t;

/**
 * Initializes all fields in a structure used to represent a linear layer.
 * Pointer for weights and biases will have the right values after the execution of this function.
 * @param layer A pointer to a structure representing a linear layer.
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Nothing is returned.
 */
void initLinear(linear_t* layer, unsigned int layer_num);

/**
 * Deallocates memory for weights and sets all pointers to NULL
 * @param layer A pointer to a structure representing a linear layer.
 * @return Nothing is returned.
 */
void clearLinear(linear_t* layer);

#ifdef __cplusplus
}
#endif

#endif