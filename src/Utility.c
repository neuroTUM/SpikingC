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
            else if(offset_type == 'S')
                offset += (layer_size[i + 1] / (sizeof(spike_t) * 8)) + 1;            
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

spike_t* returnSpikePtr(unsigned int layer_num){
    return &spike_memory[getOffset(layer_num, 'S', "LIF")];
}

void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, cfloat_array_t* In, cfloat_array_t* Out){

    if((W->cols != In->size) || (W->rows != B->size) || (Out->size != B->size)){
        printf("matrixVectorMul : Inappropriate dimensions\n"); 
        exit(1);        
    }

    cfloat_t r;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        for(unsigned int j = 0; j < W->cols; j++){
            r += W->ptr[i][j] * In->ptr[j];
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}

void matrixVectorMulSparse(wfloat_2d_array_t* W, wfloat_array_t* B, spike_array_t* In, cfloat_array_t* Out){

    if((W->cols != In->size) || (W->rows != B->size) || (Out->size != B->size)){
        printf("matrixVectorMulSparse : Inappropriate dimensions\n"); 
        exit(1);        
    }

    cfloat_t r;
    spike_t val;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        for(unsigned int j = 0; j < W->cols; j += sizeof(spike_t) * 8){
            val = In->ptr[j / (sizeof(spike_t) * 8)];
            if(val == 0)
                continue;
            else{
                for(unsigned int k = j; k < j + sizeof(spike_t) * 8 && j < W->cols; k++){
                    if(BITVALUE(val, k - j))
                        r += W->ptr[i][k];
                }
            } 
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}

void loadCSVToStaticWeightArray(const char *filepath, wfloat_t *W, unsigned int startIdx, unsigned int elements)
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

void loadCSVToStaticBiasArray(const char *filepath, wfloat_t *B, unsigned int startIdx, unsigned int size)
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

// Function to compare two arrays of floats
int compareOutputs(float *computed, float *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(computed[i] - expected[i]) > EPSILON)
        {
            return i; // Return the index of first mismatch
        }
    }
    return -1; // No mismatch found
}

void freeCSVData(float **data, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(data[i]);
    }
    free(data);
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

void loadTimestepFromFile(FILE *file, cfloat_t *scratchpadMemory, size_t timestepIndex)
{
    size_t offset = timestepIndex * INPUT_STEP_SIZE;
    fseek(file, offset, SEEK_SET);

    int16_t temp;
    for (size_t i = 0; i < DATA_POINTS_PER_TIMESTEP; ++i)
    {
        if (fread(&temp, sizeof(int16_t), 1, file) == 1)
        {
            scratchpadMemory[i] = (cfloat_t)temp;
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

void clearScratchpadMemory(cfloat_t *scratchpadMemory)
{
    // If the scratchpadMemory was dynamically allocated:
    free(scratchpadMemory);
    scratchpadMemory = NULL; // Set pointer to NULL after freeing
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
