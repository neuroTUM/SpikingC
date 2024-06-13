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
 * Loads weight values from a binary file into a static array.
 * Directly reads binary floating point data into the weight array, which is more efficient than parsing text.
 * Assumes that the binary file contains data in the format of 'float' which is then converted to 'double' if necessary.
 *
 * @param filepath Path to the binary file containing weight values.
 * @param W Pointer to the array where weights will be stored.
 * @param startIdx Starting index in the array where weights will be loaded.
 * @param elements Total number of weight elements to read from the file.
 */
void loadBinaryToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements);

/**
 * Loads bias values from a binary file into a static array.
 * Works similarly to the weight-loading function but for biases, reading directly from binary format for efficiency.
 *
 * @param filepath Path to the binary file containing bias values.
 * @param B Pointer to the array where biases will be stored.
 * @param startIdx Starting index in the array where biases will be loaded.
 * @param size Total number of bias elements to read from the file.
 */
void loadBinaryToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size);

/**
 * Initiates the loading of all weights and biases for the neural network from binary files.
 * This function manages the order and indexing of loading operations for weights and biases across multiple layers,
 * leveraging binary file operations for optimal performance.
 */
void loadBinaryStaticWeightsAndBiases();

/**
 * Loads input data from a binary file directly into the provided memory buffer.
 * Assumes the binary file contains float data (as used by your model).
 *
 * @param filename The path to the binary input file.
 * @param buffer The memory buffer to load data into.
 * @param size The number of float elements expected in the buffer.
 * @return 1 on success, 0 on failure.
 */
int loadBinaryInputData(const char *filename, cfloat_t *buffer, size_t size);

/**
 * Loads an array of floating-point data from a binary file. This function is designed to be used
 * for reading continuous blocks of float data, such as neural network weights or any floating-point parameters.
 *
 * @param filename The path to the binary file from which to read the float data.
 * @param size The number of float elements to be read from the file.
 * @return Pointer to the array of floats read from the file, or NULL if an error occurs.
 *         The caller is responsible for freeing this memory.
 * @note This function allocates memory for the returned array and the caller must free it.
 */
float *loadBinaryFloatData(const char *filename, size_t size);

/**
 * Loads an array of spike data from a binary file. This function is particularly useful
 * for neural network models where spike information is stored in a compact binary format.
 *
 * @param filename The path to the binary file from which to read the spike data.
 * @param size The number of spike_t elements to be read from the file.
 * @return Pointer to the array of spike_t data read from the file, or NULL if an error occurs.
 *         The caller is responsible for freeing this memory.
 * @note This function allocates memory for the returned array and the caller must free it.
 */
spike_t *loadBinarySpikeData(const char *filename, size_t size);

/**
 * Extracts a label from a filename string based on a naming convention where the label is embedded between underscores
 * and a dot. Example: "image_12345_9.bin" would return 9 as the label.
 *
 * @param filename The filename from which to extract the label.
 * @return The extracted label as an integer.
 */
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

/**
 * Reads input data for a specific timestep from a binary file and loads it into scratchpad memory.
 * This function seeks to the correct position in the file based on the timestep index and reads the data directly into
 * memory.
 *
 * @param file Pointer to an already opened binary file.
 * @param timestepIndex Index of the timestep to load.
 */
void loadTimestepFromFile(FILE *file, size_t timestepIndex);

#ifdef __cplusplus
}
#endif

#endif
