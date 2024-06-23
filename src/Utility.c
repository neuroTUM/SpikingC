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

fxp8_t** returnWeightPtr(unsigned int layer_num){
    fxp8_t** ptr;
    ptr = (fxp8_t **)malloc(sizeof(fxp8_t*) * layer_size[layer_num]);

    for(unsigned int i = 0; i < layer_size[layer_num]; i++){
        ptr[i] = &W[i * layer_size[layer_num + 1] + getOffset(layer_num, 'M', "Linear")];
    }
    return ptr;
}

fxp8_t* returnBiasPtr(unsigned int layer_num){
    return &B[getOffset(layer_num, 'V', "Linear")];
}

fxp16_t* returnMemPotentialPtr(unsigned int layer_num){
    return &mem_potential[getOffset(layer_num, 'V', "LIF")];
}

bool pushToList(unsigned int el){
    if(event_list == NULL){
        event_list = (event_t*)malloc(sizeof(event_t));
        if(event_list == NULL)
            return false;
        event_list->position = el;
        event_list->next = NULL;
        return true;
    }
    else{
        event_t* event = (event_t*)malloc(sizeof(event_t));
        if(event == NULL)
            return false;
        event->next = event_list;
        event->position = el;
        event_list = event;
        return true;
    }
}

void emptyList(){
    event_t *current = event_list;
    event_t *next;
    while (current != NULL)
    {
       next = current->next;
       free(current);
       current = next;
    }
    event_list = NULL;
}

#ifndef BINARY_IMPLEMENTATION

void loadCSVToStaticWeightArray(const char *filepath, fxp8_t *W, unsigned int startIdx, unsigned int elements)
{
    FILE *file = fopen(filepath, "r");
    if (!file)
    {
        perror("Failed to open file");
        return;
    }

    char buffer[BUFFER_SIZE];
    unsigned int count = 0;
    while (fgets(buffer, BUFFER_SIZE, file) && count < elements)
    {
        char *token = strtok(buffer, ",");
        while (token != NULL && count < elements)
        {
            W[startIdx + count++] = simple_atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
}

void loadCSVToStaticBiasArray(const char *filepath, fxp8_t *B, unsigned int startIdx, unsigned int size)
{
    FILE *file = fopen(filepath, "r");
    if (!file)
    {
        perror("Failed to open file");
        return;
    }

    char buffer[BUFFER_SIZE];
    unsigned int index = 0;
    while (fgets(buffer, BUFFER_SIZE, file) && index < size)
    {
        B[startIdx + index++] = simple_atof(buffer);
    }
    fclose(file);
}

void loadStaticWeightsAndBiases()
{
    // Adjust file paths and array indices as needed
    loadCSVToStaticWeightArray(PATH_WEIGHTS_FC1, W, 0, INPUT_SIZE * L1_SIZE_OUT);
    loadCSVToStaticBiasArray(PATH_BIAS_FC1, B, 0, L1_SIZE_OUT);

    // Calculate start index for each subsequent layer based on the previous layers' sizes
    unsigned int wIdx2 = INPUT_SIZE * L1_SIZE_OUT;
    unsigned int bIdx2 = L1_SIZE_OUT;

    loadCSVToStaticWeightArray(PATH_WEIGHTS_FC2, W, wIdx2, LIF1_SIZE * L2_SIZE_OUT);
    loadCSVToStaticBiasArray(PATH_BIAS_FC2, B, bIdx2, L2_SIZE_OUT);

    // And so on for each layer, adjusting the indices accordingly
    unsigned int wIdx3 = wIdx2 + LIF1_SIZE * L2_SIZE_OUT;
    unsigned int bIdx3 = bIdx2 + L2_SIZE_OUT;

    loadCSVToStaticWeightArray(PATH_WEIGHTS_FC3, W, wIdx3, LIF2_SIZE * L3_SIZE_OUT);
    loadCSVToStaticBiasArray(PATH_BIAS_FC3, B, bIdx3, L3_SIZE_OUT);
}

// Function to read CSV file into a 2D array
float **readCSV(const char *filename, int *rows, int *cols)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    if (!fgets(line, MAX_LINE_LENGTH, file))
    {
        // Handle error or empty file
        fclose(file);
        return NULL;
    }

    // Temporarily count columns
    int colCount = 1; // Starting at one since counting separators
    for (char *temp = line; *temp; temp++)
    {
        if (*temp == ',')
            colCount++;
    }

    *cols = colCount; // Assuming cols is correctly passed in

    float **data = NULL;
    *rows = 0;

    rewind(file); // Go back to start to read data again

    while (fgets(line, MAX_LINE_LENGTH, file))
    {
        data = (float **)realloc(data, (*rows + 1) * sizeof(float *));
        if (!data)
        {
            // Handle realloc failure
            *rows = 0;
            fclose(file);
            return NULL;
        }

        data[*rows] = (float*)malloc(colCount * sizeof(float));
        if (!data[*rows])
        {
            // Handle malloc failure, cleanup previously allocated rows
            for (int i = 0; i < *rows; i++)
                free(data[i]);
            free(data);
            *rows = 0;
            fclose(file);
            return NULL;
        }

        // Split line into tokens and convert to float
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < colCount)
        {
            data[*rows][col++] = simple_atof(token);
            token = strtok(NULL, ",");
        }
        (*rows)++;
    }

    fclose(file);
    return data;
}

void freeCSVData(float **data, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(data[i]);
    }
    free(data);
}

double simple_atof(const char *str)
{
    double value = 0;
    int sign = 1;

    // Skip whitespace
    while (isspace(*str))
        str++;

    // Check for sign
    if (*str == '+' || *str == '-')
    {
        sign = (*str == '-') ? -1 : 1;
        str++;
    }

    // Convert integer part
    while (isdigit(*str))
    {
        value = value * 10.0 + (*str - '0');
        str++;
    }

    // Convert fractional part
    if (*str == '.')
    {
        double fraction = 0.1;
        str++;
        while (isdigit(*str))
        {
            value += (*str - '0') * fraction;
            fraction *= 0.1;
            str++;
        }
    }

    return value * sign;
}

#else

void loadBinaryToStaticWeightArray(const char *filepath, fxp8_t *W, unsigned int startIdx, unsigned int elements)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading weights");
        return;
    }

    // Allocate a temporary buffer to hold float values
    float *tempBuffer = (float *)malloc(elements * sizeof(float));
    if (tempBuffer == NULL)
    {
        fprintf(stderr, "Memory allocation failed for temporary buffer\n");
        fclose(file);
        return;
    }

    // Read the binary data into the temporary buffer
    size_t readItems = fread(tempBuffer, sizeof(float), elements, file);
    if (readItems != elements)
    {
        fprintf(stderr, "Failed to read the expected number of weight elements from %s\n", filepath);
    }
    else
    {
        // Convert from float to double and assign to the target array
        for (unsigned int i = 0; i < elements; i++)
        {
            W[startIdx + i] = (fxp8_t)tempBuffer[i];
        }
    }

    // Free the temporary buffer and close the file
    free(tempBuffer);
    fclose(file);
}

void loadBinaryToStaticBiasArray(const char *filepath, fxp8_t *B, unsigned int startIdx, unsigned int size)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading biases");
        return;
    }

    float *tempBuffer = (float *)malloc(size * sizeof(float));
    if (tempBuffer == NULL)
    {
        fprintf(stderr, "Memory allocation failed for temporary buffer\n");
        fclose(file);
        return;
    }

    size_t readItems = fread(tempBuffer, sizeof(float), size, file);
    if (readItems != size)
    {
        fprintf(stderr, "Failed to read the expected number of bias elements from %s\n", filepath);
    }
    else
    {
        for (unsigned int i = 0; i < size; i++)
        {
            B[startIdx + i] = (fxp8_t)tempBuffer[i];
        }
    }

    free(tempBuffer);
    fclose(file);
}

void loadBinaryStaticWeightsAndBiases()
{
    // Load FC1 weights and biases
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC1_BIN, W, 0, INPUT_SIZE * L1_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC1_BIN, B, 0, L1_SIZE_OUT);

    #ifdef PRINT_WnB
    printWeightsMatrix(W, INPUT_SIZE, L1_SIZE_OUT);
    printBiasVector(B, L1_SIZE_OUT);
    #endif

    // Load FC2 weights and biases
    unsigned int wIdx2 = INPUT_SIZE * L1_SIZE_OUT;
    unsigned int bIdx2 = L1_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC2_BIN, W, wIdx2, LIF1_SIZE * L2_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC2_BIN, B, bIdx2, L2_SIZE_OUT);

    #ifdef PRINT_WnB
    printWeightsMatrix(W + wIdx2, LIF1_SIZE, L2_SIZE_OUT);
    printBiasVector(B + bIdx2, L2_SIZE_OUT);
    #endif

    // Load FC3 weights and biases
    unsigned int wIdx3 = wIdx2 + LIF1_SIZE * L2_SIZE_OUT;
    unsigned int bIdx3 = bIdx2 + L2_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC3_BIN, W, wIdx3, LIF2_SIZE * L3_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC3_BIN, B, bIdx3, L3_SIZE_OUT);

    #ifdef PRINT_WnB
    printWeightsMatrix(W + wIdx3, LIF2_SIZE, L3_SIZE_OUT);
    printBiasVector(B + bIdx3, L3_SIZE_OUT);
    #endif
}

int loadBinaryInputData(const char *filename, fxp16_t *buffer, size_t size)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("Failed to open file");
        return 0;
    }

    size_t items_read = fread(buffer, sizeof(fxp16_t), size, file);
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

#endif

#ifdef DATALOADER

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

void loadInputsFromFile(const char *filePath, fxp16_t *scratchpadMemory, size_t bufferSize)
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
    // Iterate over the buffer size, reading int16_t values and converting them to fxp16_t
    for (size_t i = 0; i < bufferSize; ++i)
    {
        if (fread(&temp, sizeof(int16_t), 1, file) == 1)
        {
            // Convert and store in the scratchpad memory
            scratchpadMemory[i] = (fxp16_t)temp;
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

void loadTimestepFromFile(FILE *file, fxp16_t *scratchpadMemory, size_t timestepIndex)
{
    size_t offset = timestepIndex * INPUT_STEP_SIZE;
    fseek(file, offset, SEEK_SET);

    int16_t temp;
    for (size_t i = 0; i < DATA_POINTS_PER_TIMESTEP; ++i)
    {
        if (fread(&temp, sizeof(int16_t), 1, file) == 1)
        {
            scratchpadMemory[i] = (fxp16_t)temp;
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

#endif
