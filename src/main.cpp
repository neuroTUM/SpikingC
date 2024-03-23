#include <iostream>
#include <chrono>
#include "../include/Model.hpp"
#include "../include/Utility.hpp"

#define TIME_STEPS 30
#define DATA_TEST_DIRECTORY "/home/copparihollmann/neuroTUM/branch_aleksav2/SpikingCpp/data/NMNIST"

#define TESTLOADER
//#define DEBUG

using namespace std;

namespace fs = std::filesystem;

int main()
{   
    auto start = chrono::high_resolution_clock::now();

    #ifdef LOAD
    loadWeights();
    #endif

    /************************************************************************************************/
    Model SNN;
    cfloat_array_t In;

    #ifdef DEBUG

    #ifdef LOAD
    parseReferenceJSON();
    #endif
    /* Reset membrane potentials and spikes */
    SNN.resetState();
    
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        
        /* Load intput for this time step */
        for(unsigned int j = 0; j < INPUT_SIZE; j++){
            scrachpad_memory[j] = output_ref["inputs"][i][0][j];
        }
        In.ptr = scrachpad_memory;
        In.size = INPUT_SIZE;

        /* Perform computations */
        SNN.run(&In);
    }

    cout << "Predicted class is " << SNN.predict() << endl;

    /************************************************************************************************/
    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    time_taken *= 1e-9;

    cout << "Time taken by program is : " << fixed << time_taken << setprecision(9);
    cout << " sec" << endl;
    #endif

    /************************************************************************************************/


    #ifdef TESTLOADER
    // Current file being processed
    unsigned int currentFileNumber = 0;
    unsigned int totalPredictions = 0;
    unsigned int correctPredictions = 0;

    // Determine the total number of .bin files
    unsigned int totalFiles = 0;
    for (const auto &entry : fs::directory_iterator(DATA_TEST_DIRECTORY))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            totalFiles++;
        }
    }

    for (const auto &entry : fs::directory_iterator(DATA_TEST_DIRECTORY))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            
            std::string filePath = entry.path().string();
            int trueLabel = extractLabelFromFilename(entry.path().filename().string());

            std::ifstream file(filePath, std::ios::binary);
            if (!file.is_open())
            {
                std::cerr << "Could not open the file " << filePath << std::endl;
            }

            SNN.resetState();

            for (size_t t = 0; t < TIME_STEPS; ++t)
            {
                std::vector<int16_t> buffer(INPUT_SIZE);
                file.read(reinterpret_cast<char *>(buffer.data()), INPUT_SIZE * sizeof(int16_t));
                for (size_t i = 0; i < INPUT_SIZE; ++i)
                {
                    scrachpad_memory[i] = static_cast<cfloat_t>(buffer[i]);
                }

                In.ptr = scrachpad_memory;
                In.size = INPUT_SIZE;

                /* Perform computations */
                SNN.run(&In);
            }

            int predictedLabel = SNN.predict();

            if (predictedLabel == trueLabel) correctPredictions++;
            totalPredictions++;

            cout << "Predicted class is: " << predictedLabel << " // LABEL: " << trueLabel << endl;

            currentFileNumber++;
            std::cout << "Processing input " << currentFileNumber << " out of " << totalFiles << std::endl;

            /************************************************************************************************/
            
        }
        // print how many files are left to process
        // Print the current progress
        
    }

    if (totalPredictions > 0)
    {
        float accuracy = static_cast<float>(correctPredictions) / totalPredictions;
        std::cout << "Accuracy: " << accuracy << std::endl;
    }
    else
    {
        std::cout << "No files processed." << std::endl;
    }

    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    time_taken *= 1e-9;

    cout << "Time taken by program is : " << fixed << time_taken << setprecision(9);
    cout << " sec" << endl;
    #endif
      

    return 0;
}