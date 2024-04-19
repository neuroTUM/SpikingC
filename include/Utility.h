#ifndef UTILITY_H
#define UTILITY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "SNNconfig.h"
#include <ctype.h>
#include <math.h>

#define MAX_LINE_LENGTH 10000 // Adjust based on your CSV line length
#define EPSILON 0.001         // Acceptable margin of error for comparison

/* Macro function used to access individual bits */
#define BITVALUE(X, N) (((X) >> (N)) & 0x1)

/**
 * Returns offsets for pointer positioning.
 * Offset type is 'M' for matrices, 'V' for float vectors and 'S' for spikes.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @param offsetType A char taking value from the following set: {'M', 'V', 'S'}.
 * @param layerType A string specifying the layer's type.
 * @return Returns the offset.
 */
unsigned int getOffset(unsigned int layer_num, char offsetType, const char* str);

/**
 * Returns an array of pointers to different rows of the weight matrix for a particular layer.
 * This function is used set the pointer to the right position in the linearized weight array.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the first element in the weight matrix for the given layer.
 */
wfloat_t** returnWeightPtr(unsigned int layer_num);

/**
 * Returns a pointer to the bias vector for a particular layer.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the first element in the bias vector for the given layer.
 */
wfloat_t* returnBiasPtr(unsigned int layer_num);

/**
 * Returns a pointer to the vector of membrane potentials for a particular layer.
 * Membrane potentials are all stored in a statically allocated array. Therefore, addresses of the first membrane potential for each layer have to be calculated and stored in a pointer used by that layer.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the first element in the array of membrane potentials for the given layer.
 */
cfloat_t* returnMemPotentialPtr(unsigned int layer_num);

/**
 * Returns a pointer to the vector of spikes for a particular layer.
 * Spikes are stored as 1-bit values inside a bigger data type. Hence accessing individual spikes requires additional functions.
 * This function is responsible for finding the first element and returning its address. The first element is always to be found at the LSB position of the underlaying data type used for storing spikes.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the array element containing the first spike value for the given layer.
 */
spike_t* returnSpikePtr(unsigned int layer_num);

/**
 * Performs matrix vector multiplication assuming floating point representation for both matrices.
 * This is necessary for the first layer because the inputs are not necessaraly ones and zeros.
 * 
 * @param W A structure containing the pointer to the first element of the weight matrix used for this layer and its dimensions.
 * @param B A structure containing the pointer to the first element of the bias vector used for this layer and its dimensions.
 * @param In A structure containing the pointer to the first element of the input vector consisting of floats used for this layer and its dimensions. This is usually the input to the whole network.
 * @param Out A structure containing the pointer to the first element of the array where outputs will be written.
 * @return Nothing is returned.
 */
void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, cfloat_array_t* In, cfloat_array_t* Out);

/**
 * Performs matrix vector multiplication assuming floating point representation for weights and binary for spikes.
 * This function is an optimized version of the standard matrix-vector multiplication because it takes input sparsity into account.
 * If all bits in the element containing spike events are zero then the corresponding computations can be skipped.
 * 
 * @param W A structure containing the pointer to the first element of the weight matrix used for this layer and its dimensions.
 * @param B A structure containing the pointer to the first element of the bias vector used for this layer and its dimensions.
 * @param In A structure containing the pointer to the first element of the input vector consisting of spike events used for this layer and its dimensions.
 * @param Out A structure containing the pointer to the first element of the array where outputs will be written.
 * @return Nothing is returned.
 */
void matrixVectorMulSparse(wfloat_2d_array_t* W, wfloat_array_t* B, spike_array_t* In, cfloat_array_t* Out);

void loadCSVToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements);

void loadCSVToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size);

void loadStaticWeightsAndBiases();

float **readCSV(const char *filename, int *rows, int *cols);

int compareOutputs(float *computed, float *expected, int size);

void freeCSVData(float **data, int rows);

int extractLabelFromFilename(const char *filename);

/**
 * Reads input data from a binary file and stores it into the scratchpad memory.
 * Assumes the binary file contains int16_t data to be converted to cfloat_t.
 *
 * @param filePath Path to the binary file to read.
 * @param scratchpadMemory Pointer to the scratchpad memory where input data should be stored.
 * @param bufferSize Number of elements to read into the scratchpad memory.
 */
void loadInputsFromFile(const char *filePath, cfloat_t *scratchpadMemory, size_t bufferSize);

double simple_atof(const char *str);

#ifdef __cplusplus
}
#endif

#endif
