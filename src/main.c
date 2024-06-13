#include "../include/Utility.h"

#define PATH_BIN_DATA "/home/aleksa_tum/main/neuroTUM/Cpp_SNN_framework/SpikingCpp/tests/NMNIST_testset_bin"

void matrixVectorMul(wfloat_2d_array_t* W, wfloat_array_t* B, spike_array_t* In, cfloat_array_t* Out){

    cfloat_t r;
    for(unsigned int i = 0; i < W->rows; i++){
        r = 0;
        for(unsigned int j = 0; j < W->cols; j++){
            r += W->ptr[i][j] * In->ptr[j];
        }
        Out->ptr[i] = r + B->ptr[i];
    }
}

void LIF(cfloat_array_t* In, spike_array_t* Out, cfloat_array_t* U, unsigned int layer_num){

    spike_t spike;
    for(unsigned int i = 0; i < U->size; i++){
        if(U->ptr[i] > threshold[layer_num])
            spike = 1;
        else
            spike = 0;

        U->ptr[i] = Beta[layer_num] * (U->ptr[i] - spike * threshold[layer_num]) + In->ptr[i];
        Out->ptr[i] = spike;
    }
}

int main(void)
{

    // Load weights and biases
    loadBinaryStaticWeightsAndBiases();

    // Weights and biases
    wfloat_2d_array_t W1;
    W1.ptr = returnWeightPtr(0);
    W1.rows = layer_size[1];
    W1.cols = layer_size[0];
    wfloat_array_t B1;
    B1.ptr = returnBiasPtr(0);
    B1.size	= layer_size[1];
    wfloat_2d_array_t W2;
    W2.ptr = returnWeightPtr(2);
    W2.rows = layer_size[3];
    W2.cols = layer_size[2];
    wfloat_array_t B2;
    B2.ptr = returnBiasPtr(2);
    B2.size	= layer_size[3];
    wfloat_2d_array_t W3;
    W3.ptr = returnWeightPtr(4);
    W3.rows = layer_size[5];
    W3.cols = layer_size[4];
    wfloat_array_t B3;
    B3.ptr = returnBiasPtr(4);
    B3.size	= layer_size[5];

    // Membrane potentials
    cfloat_array_t U1;
    U1.ptr = returnMemPotentialPtr(1);
    U1.size = layer_size[1];
    cfloat_array_t U2;
    U2.ptr = returnMemPotentialPtr(1);
    U2.size = layer_size[1];
    cfloat_array_t U3;
    U3.ptr = returnMemPotentialPtr(1);
    U3.size = layer_size[1];

    // Array used for storing prediction results
    unsigned int actPred[LIF3_SIZE];

    // Intermediate results for LIF and linear layers
    spike_array_t Spikes;
    Spikes.size = INPUT_SIZE;
    Spikes.ptr = scrachpad_memory_spikes;

    cfloat_array_t MatrixResult;
    MatrixResult.size = L1_SIZE_OUT;
    MatrixResult.ptr = scrachpad_memory_floats;

    const char *dataTestDirectory = PATH_BIN_DATA;
    DIR *dir;
    struct dirent *entry;

    unsigned int totalPredictions = 0;
    unsigned int correctPredictions = 0;

    if ((dir = opendir(dataTestDirectory)) == NULL)
    {
        perror("Failed to open directory");
        return -1;
    }

    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG && strstr(entry->d_name, ".bin") != NULL)
        {
            char filePath[1024];
            snprintf(filePath, sizeof(filePath), "%s/%s", dataTestDirectory, entry->d_name);

            int trueLabel = extractLabelFromFilename(entry->d_name);

            FILE *file = fopen(filePath, "rb");
            if (!file)
            {
                perror("Failed to open file");
                return -1;
            }  

            // Reset for membrane potentials, spikes and prediction results
            for(unsigned int i = 0; i < INPUT_SIZE; i++)
                scrachpad_memory_spikes[i] = 0;

            for(unsigned int i = 0; i < L1_SIZE_OUT; i++)
                scrachpad_memory_floats[i] = 0;

            for(unsigned int i = 0; i < LIF3_SIZE; i++)
                actPred[i] = 0;                

            for(unsigned int i = 0; i < TIME_STEPS; i++)
            {
                Spikes.size = INPUT_SIZE;
                loadTimestepFromFile(file, i);

                // Linear layer 1
                MatrixResult.size = L1_SIZE_OUT;
                matrixVectorMul(&W1, &B1, &Spikes, &MatrixResult);
                // LIF layer 1
                Spikes.size = LIF1_SIZE;
                LIF(&MatrixResult, &Spikes, &U1, 1);
                // Linear layer 2
                MatrixResult.size = L2_SIZE_OUT;
                matrixVectorMul(&W2, &B2, &Spikes, &MatrixResult);
                // LIF layer 2
                Spikes.size = LIF2_SIZE;
                LIF(&MatrixResult, &Spikes, &U2, 3);
                // Linear layer 3
                MatrixResult.size = L3_SIZE_OUT;
                matrixVectorMul(&W3, &B3, &Spikes, &MatrixResult);
                // LIF layer 3
                Spikes.size = LIF3_SIZE;
                LIF(&MatrixResult, &Spikes, &U3, 5);

                // Accumulating prediction results
                for(unsigned int i = 0; i < LIF3_SIZE; i++)
                    actPred[i] += Spikes.ptr[i];
            }

            int predictedLabel = 0;
            unsigned int max = actPred[0];
            for(unsigned int i = 1; i < LIF3_SIZE; i++){
                if(actPred[i] > max){
                    predictedLabel = i;
                    max = actPred[i];
                }
            }

            if (predictedLabel == trueLabel)
            {
                correctPredictions++;
            }
            totalPredictions++;

            printf("Processed %s: Predicted class = %d, True Label = %d\n", entry->d_name, predictedLabel, trueLabel);
            
            fclose(file);
        }
    }

    closedir(dir);

    if (totalPredictions > 0)
    {
        float accuracy = (float)correctPredictions / totalPredictions;
        printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
    else
    {
        printf("No .bin files processed.\n");
    }

    return 0;
}
