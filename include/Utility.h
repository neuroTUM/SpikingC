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

#ifndef BINARY_IMPLEMENTATION

/**
 * Loads weight values from a CSV file into a static array for neural network weights.
 * This function parses the CSV file and converts string representations of floating point numbers into actual floating
 * point values, storing them sequentially into a provided array starting at a specified index.
 *
 * @param filepath Path to the CSV file containing weight values.
 * @param W Pointer to the array where weights will be stored.
 * @param startIdx Starting index in the array where weights will be loaded.
 * @param elements Total number of weight elements to read from the file.
 */
void loadCSVToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements);

/**
 * Loads bias values from a CSV file into a static array.
 * Similar to the weight-loading function but specifically designed for bias values,
 * reading a single line of floating point numbers representing biases for a network layer.
 *
 * @param filepath Path to the CSV file containing bias values.
 * @param B Pointer to the array where biases will be stored.
 * @param startIdx Starting index in the array where biases will be loaded.
 * @param size Total number of bias elements to read from the file.
 */
void loadCSVToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size);

/**
 * Initiates the loading of all weights and biases for the neural network from CSV files.
 * This function orchestrates the sequential loading of weights and biases for multiple layers of a neural network,
 * setting the correct indices and sizes for each layer based on predefined layer dimensions.
 */
void loadStaticWeightsAndBiases();

/**
 * Reads a CSV file and stores its contents into a dynamically allocated 2D array of floats.
 * This function opens a CSV file, reads it line by line, and splits each line into tokens based on comma delimiters,
 * converting each token into a float and storing it in an array.
 *
 * @param filename Path to the CSV file.
 * @param rows Pointer to an integer where the number of rows will be stored.
 * @param cols Pointer to an integer where the number of columns will be stored.
 * @return A pointer to a 2D array of floats containing the parsed data.
 */
float **readCSV(const char *filename, int *rows, int *cols);

/**
 * Frees the memory allocated for a 2D array of floats.
 * This function is used to clean up memory after it is no longer needed, typically after processing CSV data.
 *
 * @param data Pointer to the 2D array of floats.
 * @param rows Number of rows in the array (i.e., how many pointers in the first dimension to free).
 */
void freeCSVData(float **data, int rows);

/**
 * Converts a string to a floating point number (double precision).
 * This function is a simple alternative to standard library functions like atof, providing error handling and improved
 * robustness.
 *
 * @param str Pointer to the string containing the number to convert.
 * @return The converted double value.
 */
double simple_atof(const char *str);

#else

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

#endif

#ifdef PRINT_WnB

/**
 * Prints a matrix of weights to the console for debugging or analysis purposes.
 * This function iterates over rows and columns of the weight matrix, printing each element in a formatted manner.
 *
 * @param W Pointer to the weight matrix.
 * @param rows Number of rows in the weight matrix.
 * @param cols Number of columns in the weight matrix.
 */
void printWeightsMatrix(wfloat_t *W, unsigned int rows, unsigned int cols);

/**
 * Prints a vector of biases to the console.
 * Each bias is printed on a new line, with formatting to ensure clarity and ease of reading.
 *
 * @param B Pointer to the bias vector.
 * @param size Number of elements in the bias vector.
 */
void printBiasVector(wfloat_t *B, unsigned int size);

#endif

#ifdef DATALOADER

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
 * @param scratchpadMemory Pointer to the scratchpad memory where input data for the timestep should be stored.
 * @param timestepIndex Index of the timestep to load.
 */
void loadTimestepFromFile(FILE *file, cfloat_t *scratchpadMemory, size_t timestepIndex);

#endif

#ifdef __cplusplus
}
#endif

#endif
