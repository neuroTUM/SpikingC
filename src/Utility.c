#include "../include/Utility.h"

#define BUFFER_SIZE 30055

unsigned int getOffset(unsigned int layer_num, char offset_type, const char* str){
    unsigned int offset = 0;

    for(unsigned int i = 0; i < layer_num; i++){
        if(strcmp(layer_type[i], "Linear") == 0 && strcmp(str, "Linear") == 0){
            if(offset_type == 'M')
                offset += layer_size[i] * layer_size[i + 1];
            else if(offset_type == 'V')
                offset += layer_size[i + 1];
        }
        else if(strcmp(layer_type[i], "LIF") == 0 && strcmp(str, "LIF") == 0){
            if(offset_type == 'V')
                offset += layer_size[i + 1];            
        }
    }

    return offset;
}

wfloat_t** returnWeightPtr(unsigned int layer_num){
    wfloat_t** ptr;
    ptr = (wfloat_t **)malloc(sizeof(wfloat_t*) * layer_size[layer_num + 1]);

    for(unsigned int i = 0; i < layer_size[layer_num + 1]; i++){
        ptr[i] = &W[i * layer_size[layer_num] + getOffset(layer_num, 'M', "Linear")];
    }
    return ptr;
}

wfloat_t* returnBiasPtr(unsigned int layer_num){
    return &B[getOffset(layer_num, 'V', "Linear")];
}

cfloat_t* returnMemPotentialPtr(unsigned int layer_num){
    return &mem_potential[getOffset(layer_num, 'V', "LIF")];
}

void loadBinaryToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading weights");
        return;
    }

    // Read the binary data into the temporary buffer
    size_t readItems = fread(W + startIdx, sizeof(float), elements, file);
    if (readItems != elements)
    {
        fprintf(stderr, "Failed to read the expected number of weight elements from %s\n", filepath);
    }

    fclose(file);
}

void loadBinaryToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading biases");
        return;
    }

    size_t readItems = fread(B + startIdx, sizeof(float), size, file);
    if (readItems != size)
    {
        fprintf(stderr, "Failed to read the expected number of bias elements from %s\n", filepath);
    }

    fclose(file);
}

void loadBinaryStaticWeightsAndBiases()
{
    // Load FC1 weights and biases
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC1_BIN, W, 0, INPUT_SIZE * L1_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC1_BIN, B, 0, L1_SIZE_OUT);

    // Load FC2 weights and biases
    unsigned int wIdx2 = INPUT_SIZE * L1_SIZE_OUT;
    unsigned int bIdx2 = L1_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC2_BIN, W, wIdx2, LIF1_SIZE * L2_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC2_BIN, B, bIdx2, L2_SIZE_OUT);

    // Load FC3 weights and biases
    unsigned int wIdx3 = wIdx2 + LIF1_SIZE * L2_SIZE_OUT;
    unsigned int bIdx3 = bIdx2 + L2_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC3_BIN, W, wIdx3, LIF2_SIZE * L3_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC3_BIN, B, bIdx3, L3_SIZE_OUT);
}

int loadBinaryInputData(const char *filename, cfloat_t *buffer, size_t size)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Failed to open file");
        return 0;
    }

    size_t items_read = fread(buffer, sizeof(cfloat_t), size, file);
    if (items_read != size)
    {
        fprintf(stderr, "Error reading binary file: %s\n", filename);
        fclose(file);
        return 0;
    }

    fclose(file);
    return 1;
}

float *loadBinaryFloatData(const char *filename, size_t size)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Failed to open file for reading");
        return NULL;
    }

    float *data = malloc(size * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory for reading data\n");
        fclose(file);
        return NULL;
    }

    size_t items_read = fread(data, sizeof(float), size, file);
    if (items_read != size)
    {
        fprintf(stderr, "Failed to read the expected number of items from %s\n", filename);
        free(data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}

spike_t *loadBinarySpikeData(const char *filename, size_t size)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Failed to open file for reading");
        return NULL;
    }

    spike_t *data = malloc(size * sizeof(spike_t));
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory for reading data\n");
        fclose(file);
        return NULL;
    }

    size_t items_read = fread(data, sizeof(spike_t), size, file);
    if (items_read != size)
    {
        fprintf(stderr, "Failed to read the expected number of items from %s\n", filename);
        free(data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return data;
}

int extractLabelFromFilename(const char *filename)
{
    // Find the last occurrence of underscore and dot in the filename
    const char *lastUnderscore = strrchr(filename, '_');
    const char *lastDot = strrchr(filename, '.');

    if (!lastUnderscore || !lastDot)
    {
        fprintf(stderr, "Filename format error: %s\n", filename);
        return -1; 
    }

    // Calculate positions for label extraction
    size_t startPos = lastUnderscore - filename + 1;
    size_t length = lastDot - lastUnderscore - 1;

    // Extract the label substring
    char labelStr[10]; 
    strncpy(labelStr, filename + startPos, length);
    labelStr[length] = '\0'; // Null-terminate the extracted substring

    // Convert extracted label to integer
    int label = atoi(labelStr);
    return label;
}

void loadInputsFromFile(const char *filePath, cfloat_t *scratchpadMemory, size_t bufferSize)
{
    // Open the file for reading in binary mode
    FILE *file = fopen(filePath, "rb");
    if (!file)
    {
        perror("Could not open the file");
        return;
    }

    // Temporary buffer to hold the int16_t data read from the file
    int16_t temp;
    // Iterate over the buffer size, reading int16_t values and converting them to cfloat_t
    for (size_t i = 0; i < bufferSize; ++i)
    {
        if (fread(&temp, sizeof(int16_t), 1, file) == 1)
        {
            // Convert and store in the scratchpad memory
            scratchpadMemory[i] = (cfloat_t)temp;
        }
        else
        {
            // Handle reading error or EOF
            if (!feof(file))
            { // Check if the end of the file has been reached
                fprintf(stderr, "Error reading file before reaching the expected buffer size\n");
            }
            break;
        }
    }

    // Close the file
    fclose(file);
}

void loadTimestepFromFile(FILE *file, size_t timestepIndex)
{
    size_t offset = timestepIndex * INPUT_STEP_SIZE;
    fseek(file, offset, SEEK_SET);

    int16_t temp;
    for (size_t i = 0; i < DATA_POINTS_PER_TIMESTEP; ++i)
    {
        if (fread(&temp, sizeof(int16_t), 1, file) == 1)
        {
            scrachpad_memory_spikes[i] = (spike_t)temp;
        }
        else
        {
            if (!feof(file))
            {
                fprintf(stderr, "Error reading file at timestep %zu\n", timestepIndex);
            }
            break;
        }
    }
}