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

void loadBinaryToStaticWeightArray(const char *filepath, unsigned int startIdx, unsigned int elements)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading weights");
        return;
    }

    // Read the binary data into the temporary buffer
    size_t readItems = fread(W + startIdx, sizeof(fxp8_t), elements, file);
    if (readItems != elements)
    {
        fprintf(stderr, "Failed to read the expected number of weight elements from %s\n", filepath);
    }

    fclose(file);
}

void loadBinaryToStaticBiasArray(const char *filepath, unsigned int startIdx, unsigned int size)
{
    FILE *file = fopen(filepath, "rb");
    if (!file)
    {
        perror("Failed to open file for reading biases");
        return;
    }

    size_t readItems = fread(B + startIdx, sizeof(fxp8_t), size, file);
    if (readItems != size)
    {
        fprintf(stderr, "Failed to read the expected number of bias elements from %s\n", filepath);
    }

    fclose(file);
}

void loadBinaryStaticWeightsAndBiases()
{
    // Load FC1 weights and biases
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC1_BIN, 0, INPUT_SIZE * L1_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC1_BIN, 0, L1_SIZE_OUT);

    // Load FC2 weights and biases
    unsigned int wIdx2 = INPUT_SIZE * L1_SIZE_OUT;
    unsigned int bIdx2 = L1_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC2_BIN, wIdx2, LIF1_SIZE * L2_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC2_BIN, bIdx2, L2_SIZE_OUT);

    // Load FC3 weights and biases
    unsigned int wIdx3 = wIdx2 + LIF1_SIZE * L2_SIZE_OUT;
    unsigned int bIdx3 = bIdx2 + L2_SIZE_OUT;
    loadBinaryToStaticWeightArray(PATH_WEIGHTS_FC3_BIN, wIdx3, LIF2_SIZE * L3_SIZE_OUT);
    loadBinaryToStaticBiasArray(PATH_BIAS_FC3_BIN, bIdx3, L3_SIZE_OUT);
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