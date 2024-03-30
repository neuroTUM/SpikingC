#include "../include/Utility.h"

#define BUFFER_SIZE 30055

unsigned int getOffset(unsigned int layer_num, char offset_type, char* str){
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
    ptr = malloc(sizeof(wfloat_t*) * layer_size[layer_num + 1]);

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
            W[startIdx + count++] = atof(token);
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
        B[startIdx + index++] = atof(buffer);
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

    float **data = NULL;
    char line[MAX_LINE_LENGTH];
    *rows = 0;
    while (fgets(line, MAX_LINE_LENGTH, file))
    {
        data = realloc(data, (*rows + 1) * sizeof(float *));
        data[*rows] = malloc(*cols * sizeof(float)); // Assumes 'cols' is set to the correct number of columns

        // Split line into tokens and convert to float
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL)
        {
            data[*rows][col++] = atof(token);
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
