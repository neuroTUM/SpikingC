#include "../include/Utility.h"

#define PATH_BIN_DATA "/home/aleksa_tum/main/neuroTUM/Cpp_SNN_framework/SpikingCpp/tests/NMNIST_testset_bin"

void matrixVectorMulSparse(fxp8_2d_array_t* W, fxp8_array_t* B, fxp16_array_t* Out){

    event_t* temp = event_list;

    /* Initialize all elements in Out with values from B */
    for(unsigned int i = 0; i < B->size; i++){
        Out->ptr[i] = B->ptr[i];
    }

    /* Traverse through all non-zero elements */
    while(temp != NULL)
    {
        /* Add the whole columns element-wise to the output vector */
        for(unsigned int i = 0; i < W->rows; i++){

            /* IMPORTANT: All checks for over/underflow are a part of the custom instruction and shouldn't be
                        visible in the C code. Here I only want to simulate the effect those instructions will 
                        have once they are implemented.
            */            

            /* Saturate the output value if it goes outside of the predefined range */
            if((int)Out->ptr[i] + (int)W->ptr[temp->position][i] < INT16_MIN){
                Out->ptr[i] = INT16_MIN;
            }
            else if((int)Out->ptr[i] + (int)W->ptr[temp->position][i] > INT16_MAX){
                Out->ptr[i] = INT16_MAX;
            }
            else{
                Out->ptr[i] += W->ptr[temp->position][i];
            }
        }
        temp = temp->next;
    }
}

void LIF(fxp16_array_t* In, fxp16_array_t* U, unsigned int layer_num){

    bool spike;
    emptyList();
    for(unsigned int i = 0; i < U->size; i++){
        if(U->ptr[i] > threshold[layer_num]){
            spike = 1;
            pushToList(i);
        }
        else{
            spike = 0;
        }

        /* IMPORTANT: All checks for over/underflow are a part of the custom instruction and shouldn't be
                      visible in the C code. Here I only want to simulate the effect those instructions will 
                      have once they are implemented.
        */

        /* When performing the shifht right operation we don't have to worry about over/underflow */
        U->ptr[i] = (U->ptr[i] >> Beta[layer_num]);
        
        /* Check for underflow */
        if(((int)U->ptr[i] - (int)(spike * L[layer_num])) < INT16_MIN){
            U->ptr[i] = INT16_MIN;
        }
        else{
            U->ptr[i] -= spike * L[layer_num];
        }

        /* Saturate the synaptic current if it goes outside of the predefined range */
        int temp = In->ptr[i] * 2;
        if(temp < INT16_MIN){
            In->ptr[i] = INT16_MIN;
        }
        else if(temp > INT16_MAX){
            In->ptr[i] = INT16_MAX;
        }
        else{
            In->ptr[i] = In->ptr[i] * 2;
        }

        /* Saturate the membrane potential value if it goes outside of the predefined range */
        if((int)U->ptr[i] + (int)In->ptr[i] < INT16_MIN){
            U->ptr[i] = INT16_MIN;
        }
        else if((int)U -> ptr[i] + (int)In->ptr[i] > INT16_MAX){
            U->ptr[i] = INT16_MAX;
        }
        else{
            U->ptr[i] += In->ptr[i];
        }
    }
}

int main(void)
{

    // Load weights and biases
    loadStaticWeightsAndBiases();

    // Weights and biases
    fxp8_2d_array_t W1;
    W1.ptr = returnWeightPtr(0);
    W1.rows = layer_size[1];
    W1.cols = layer_size[0];
    fxp8_array_t B1;
    B1.ptr = returnBiasPtr(0);
    B1.size	= layer_size[1];
    fxp8_2d_array_t W2;
    W2.ptr = returnWeightPtr(2);
    W2.rows = layer_size[3];
    W2.cols = layer_size[2];
    fxp8_array_t B2;
    B2.ptr = returnBiasPtr(2);
    B2.size	= layer_size[3];
    fxp8_2d_array_t W3;
    W3.ptr = returnWeightPtr(4);
    W3.rows = layer_size[5];
    W3.cols = layer_size[4];
    fxp8_array_t B3;
    B3.ptr = returnBiasPtr(4);
    B3.size	= layer_size[5];

    // Membrane potentials
    fxp16_array_t U1;
    U1.ptr = returnMemPotentialPtr(1);
    U1.size = layer_size[1];
    fxp16_array_t U2;
    U2.ptr = returnMemPotentialPtr(3);
    U2.size = layer_size[3];
    fxp16_array_t U3;
    U3.ptr = returnMemPotentialPtr(5);
    U3.size = layer_size[5];

    // Matrix multiplication result
    fxp16_array_t matmul_out;
    matmul_out.ptr = scrachpad_memory;

    // Array used for storing prediction results
    unsigned int actPred[LIF3_SIZE];

    // ----------------------------------------------------------------------------------------------- //
    // ---------------------------------------- Data loading ----------------------------------------- //
    // ----------------------------------------------------------------------------------------------- //

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

            // Reset the membrane potential
            for(unsigned int i = 0; i < LIF1_SIZE + LIF2_SIZE + LIF3_SIZE; i++)
                mem_potential[i] = 0;

            // Empty the list
            emptyList();

            for(unsigned int i = 0; i < LIF3_SIZE; i++)
                actPred[i] = 0;  

            for (unsigned int i = 0; i < TIME_STEPS; i++)
            {
                loadTimestepFromFile(file, scrachpad_memory, i);

                // ----------------------------------------------------------------------------------------------- //
                // --------------------------------------- Data processing --------------------------------------- //
                // ----------------------------------------------------------------------------------------------- //

                /**********************************************************************************/
                /* This code is necessary in order to transform the input vector
                   into a list of events
                */
                for(unsigned int i = 0; i < INPUT_SIZE; i++){
                    if(scrachpad_memory[i] == 1){
                        pushToList(i);
                    }
                    else if(scrachpad_memory[i] == 2){
                        pushToList(i);
                        pushToList(i);
                    }
                }
                /**********************************************************************************/

                matmul_out.size = L1_SIZE_OUT;
                matrixVectorMulSparse(&W1, &B1, &matmul_out);
                LIF(&matmul_out, &U1, 1);
                matmul_out.size = L2_SIZE_OUT;
                matrixVectorMulSparse(&W2, &B2, &matmul_out);
                LIF(&matmul_out, &U2, 3);
                matmul_out.size = L3_SIZE_OUT;
                matrixVectorMulSparse(&W3, &B3, &matmul_out);
                LIF(&matmul_out, &U3, 5);

                event_t *current = event_list;
                while (current != NULL)
                {
                    actPred[current->position] += 1;
                    current = current->next;
                }
                emptyList();
                
            }

            // ----------------------------------------------------------------------------------------------- //
            // -------------------------------------- Making a prediction ------------------------------------ //
            // ----------------------------------------------------------------------------------------------- //

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

    free(W1.ptr);
    free(W2.ptr);
    free(W3.ptr);

    return 0;
}