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

/* Initializes pointers for weights and biases */
void initLinear(linear_t* layer, unsigned int layer_num);

/* Deallocates memory for weights and sets all pointers to NULL */
void clearLinear(linear_t* layer);

#ifdef __cplusplus
}
#endif

#endif