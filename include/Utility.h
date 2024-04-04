/*
    Author's name: Aleksa Stojkovic
    Date of creation: 27.3.2024
    Description: A set of commonly used functions(mainly matrix multiplication and pointer manipulations) 
*/

#ifndef UTILITY_H
#define UTILITY_H

#include "SNNconfig.h"
#include <math.h>

#define MAX_LINE_LENGTH 10000 // Adjust based on your CSV line length
#define EPSILON 0.001         // Acceptable margin of error for comparison

/* Macro function used to access individual bits */
#define BITVALUE(X, N) (((X) >> (N)) & 0x1)

/* Returns offsets for pointer positioning */
/* Offset type is 'M' for matrices, 'V' for float vectors and 'S' for spikes */
unsigned int getOffset(unsigned int layer_num, char offsetType, char* str);

/* Returns an array of pointers to different rows of the weight matrix for a particular layer */
wfloat_t** returnWeightPtr(unsigned int layer_num);

/* Returns a pointer to the bias vector for a particular layer */
wfloat_t* returnBiasPtr(unsigned int layer_num);

/* Returns a pointer to the vector of membrane potentials for a particular layer */
cfloat_t* returnMemPotentialPtr(unsigned int layer_num);

/* Returns a pointer to the vector of spikes for a particular layer */
spike_t* returnSpikePtr(unsigned int layer_num);

/* Performs matrix vector multiplication assuming floating point representation */
/* This is necessary for the first layer because the inputs are not just ones and zeros */
void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, cfloat_array_t* In, cfloat_array_t* Out);

/* Performs matrix vector multiplication assuming floating point representation for weights and binary for spikes */
void matrixVectorMulSparse(wfloat_2d_array_t* W, wfloat_array_t* B, spike_array_t* In, cfloat_array_t* Out);

void loadCSVToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements);

void loadCSVToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size);

void loadStaticWeightsAndBiases();

float **readCSV(const char *filename, int *rows, int *cols);

int compareOutputs(float *computed, float *expected, int size);

void freeCSVData(float **data, int rows);

int extractLabelFromFilename(const char *filename);

void loadInputsFromFile(const char *filePath, cfloat_t *scratchpadMemory, size_t bufferSize);

#endif