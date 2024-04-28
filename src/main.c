#include "../include/Model.h"
#include "../include/Utility.h"

#define PATH_BIN_DATA "/home/copparihollmann/neuroTUM/NMNIST/"

int main(void)
{
    model_t SNN;
    initModel(&SNN);

    cfloat_array_t In;
    In.size = INPUT_SIZE;
    In.ptr = scrachpad_memory;
    if (!In.ptr)
    {
        perror("Failed to allocate memory for input");
        return -1;
    }

    #ifndef BINARY_IMPLEMENTATION
    /* Load weights and biases */
    loadStaticWeightsAndBiases();
    #else
    /* Load BINARY weights and biases */
    loadBinaryStaticWeightsAndBiases();
    #endif

    /* Reset the state */
    SNN.resetState_fptr(&SNN);
    
    #ifdef TEST
    char filename[256];
    for (unsigned int i = 0; i < TIME_STEPS; i++)
    {
        #ifndef BINARY_IMPLEMENTATION
        // Construct the filename for the current timestep
        sprintf(filename, "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/input/inputs_timestep_%u.csv", i);

        // Load the data for this time step
        int rows, cols;
        float **inputData = readCSV(filename, &rows, &cols);
        if (!inputData || rows < 1)
        {
            fprintf(stderr, "Failed to load input data for timestep %u\n", i);
            exit(1);
        }

        // Assume inputData[0] contains the input for this timestep
        int cntZeors = 0;
        for (unsigned int j = 0; j < (unsigned int)cols && j < In.size; j++)
        {
            In.ptr[j] = inputData[0][j];
            if(In.ptr[j] == 0){
                cntZeors++;
            }
        }

        /* Run the model for one time step */
        SNN.run_fptr(&SNN, &In);

        freeCSVData(inputData, rows);
        #else
        // Construct the filename for the current timestep
        sprintf(filename, "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs_binary/inputs/inputs_timestep_%u.bin", i);

        // Load the data for this time step
        if (!loadBinaryInputData(filename, In.ptr, In.size))
        {
            fprintf(stderr, "Failed to load input data for timestep %u\n", i);
            exit(1);
        }
        /* Run the model for one time step */
        SNN.run_fptr(&SNN, &In);
        #endif

        
    }

    printf("Predicted class is: %d\n", SNN.predict_fptr(&SNN));


    /* Free memory allocated for SNN */
    SNN.clearModel_fptr(&SNN);
    #endif

    #ifdef DATALOADER
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

            for (unsigned int i = 0; i < TIME_STEPS; i++)
            {
                loadTimestepFromFile(file, scrachpad_memory, i);
                SNN.run_fptr(&SNN, &In);
                
            }
            
            //loadInputsFromFile(filePath, scrachpad_memory, INPUT_SIZE);

            //SNN.run_fptr(&SNN, &In);

            int predictedLabel = SNN.predict_fptr(&SNN);

            if (predictedLabel == trueLabel)
            {
                correctPredictions++;
            }
            totalPredictions++;

            printf("Processed %s: Predicted class = %d, True Label = %d\n", entry->d_name, predictedLabel, trueLabel);
            
            fclose(file);

            /* Reset the state */
            SNN.resetState_fptr(&SNN);
        }
    }

    SNN.clearModel_fptr(&SNN);

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
    #endif

    return 0;
}
