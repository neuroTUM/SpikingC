/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: A set of structures and functions used to describe and perform computations for a linear layer 
*/

#ifndef LINEAR_H
#define LINEAR_H

#include "Utility.h"

/* A structure holding all relevant data for linear layers */
typedef struct{
    unsigned int layer_num;
    wfloat_2d_array_t W;
    wfloat_array_t B;
    /* A pointer to the clearLinear function */
    void (*clearLinear_fptr) (linear_t*);
    void (*matrixVectorMul_fptr) (wfloat_2d_array_t*, wfloat_array_t*, cfloat_array_t*, cfloat_array_t*);
    void (*matrixVectorMulSparse_fptr) (wfloat_2d_array_t*, wfloat_array_t*, spike_array_t*, cfloat_array_t*);
} linear_t;

/* Initializes pointers for weights and biases */
void initLinear(linear_t* layer, unsigned int layer_num);

/* Deallocates memory for weights and sets all pointers to NULL */
void clearLinear(linear_t* layer);

#endif