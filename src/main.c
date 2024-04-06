#include "../include/Model.h"
#include "../include/Utility.h"

#define PATH_BIN_DATA "/home/copparihollmann/neuroTUM/SpikingCpp/tests/NMNIST"

int main(void)
{   

    model_t SNN;
    initModel(&SNN);

    cfloat_array_t In;
    In.size = INPUT_SIZE;
    In.ptr = scrachpad_memory; // I have the feeling that this should be declared inside of the for loop like we did in the cpp version
    if (!In.ptr)
    {
        perror("Failed to allocate memory for input");
        return -1;
    }

    /* Load weights and biases */
    loadStaticWeightsAndBiases();

    /* Reset the state */
    SNN.resetState_fptr(&SNN);

    char filename[256];
    for (unsigned int i = 0; i < TIME_STEPS; i++)
    {
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
        for (unsigned int j = 0; j < (unsigned int)cols && j < In.size; j++)
        {
            In.ptr[j] = inputData[0][j];
        }

        /* Run the model for one time step */
        SNN.run_fptr(&SNN, &In);

        freeCSVData(inputData, rows);
    }

    printf("Predicted class is: %d\n", SNN.predict_fptr(&SNN));

    /* Free memory allocated for In */
    //free(In.ptr);

    /* Free memory allocated for SNN */
    SNN.clearModel_fptr(&SNN);

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

            
            loadInputsFromFile(filePath, scrachpad_memory, INPUT_SIZE);

            cfloat_array_t In = {.ptr = scrachpad_memory, .size = INPUT_SIZE};

            SNN.run_fptr(&SNN, &In);

            int predictedLabel = SNN.predict_fptr(&SNN);

            if (predictedLabel == trueLabel)
            {
                correctPredictions++;
            }
            totalPredictions++;

            printf("Processed %s: Predicted class = %d, True Label = %d\n", entry->d_name, predictedLabel, trueLabel);
            
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
