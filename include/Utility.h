#ifndef UTILITY_H
#define UTILITY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "SNNconfig.h"
#include <ctype.h>
#include <math.h>

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
fxp8_t** returnWeightPtr(unsigned int layer_num);

/**
 * Returns a pointer to the bias vector for a particular layer.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the first element in the bias vector for the given layer.
 */
fxp8_t* returnBiasPtr(unsigned int layer_num);

/**
 * Returns a pointer to the vector of membrane potentials for a particular layer.
 * Membrane potentials are all stored in a statically allocated array. Therefore, addresses of the first membrane potential for each layer have to be calculated and stored in a pointer used by that layer.
 * 
 * @param layer_num The current layer number. The first layers is always marked with 0.
 * @return Returns a pointer pointing to the first element in the array of membrane potentials for the given layer.
 */
fxp16_t* returnMemPotentialPtr(unsigned int layer_num);

/**
 * Pushes an element to the front of a linked list.
 * 
 * @param el The value that has to be pushed
 * @return True if memory allocation was successful. Otherwise false.
*/
bool pushToList(unsigned int el);

/**
 * Empties a linked list completely.
 * 
 * @return Nothing is returned.
*/
void emptyList();

/**
 * Loads weight values from a binary file into a static array.
 * Directly reads binary floating point data into the weight array, which is more efficient than parsing text.
 * Assumes that the binary file contains data in the format of 'float' which is then converted to 'double' if necessary.
 *
 * @param filepath Path to the binary file containing weight values.
 * @param W Pointer to the array where weights will be stored.
 * @param startIdx Starting index in the array where weights will be loaded.
 * @param elements Total number of weight elements to read from the file.
 */
void loadBinaryToStaticWeightArray(const char *filepath, unsigned int startIdx, unsigned int elements);

/**
 * Loads bias values from a binary file into a static array.
 * Works similarly to the weight-loading function but for biases, reading directly from binary format for efficiency.
 *
 * @param filepath Path to the binary file containing bias values.
 * @param B Pointer to the array where biases will be stored.
 * @param startIdx Starting index in the array where biases will be loaded.
 * @param size Total number of bias elements to read from the file.
 */
void loadBinaryToStaticBiasArray(const char *filepath, unsigned int startIdx, unsigned int size);

/**
 * Initiates the loading of all weights and biases for the neural network from binary files.
 * This function manages the order and indexing of loading operations for weights and biases across multiple layers,
 * leveraging binary file operations for optimal performance.
 */
void loadBinaryStaticWeightsAndBiases();

/**
 * Reads input data for a specific timestep from a binary file and loads it into scratchpad memory.
 * This function seeks to the correct position in the file based on the timestep index and reads the data directly into
 * memory.
 *
 * @param file Pointer to an already opened binary file.
 * @param scratchpadMemory Pointer to the scratchpad memory where input data for the timestep should be stored.
 * @param timestepIndex Index of the timestep to load.
 */
void loadTimestepFromFile(FILE *file, fxp16_t *scratchpadMemory, size_t timestepIndex);

#ifdef __cplusplus
}
#endif

#endif
