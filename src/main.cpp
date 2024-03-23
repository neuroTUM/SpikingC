#include <iostream>
#include <chrono>
#include "../include/Model.hpp"

#define TIME_STEPS 30
#define DATA_TEST_DIRECTORY "/home/copparihollmann/neuroTUM/branch_aleksav2/SpikingCpp/data/NMNIST"

#define TESTLOADER

using namespace std;

namespace fs = std::filesystem;

int extractLabelFromFilename(const std::string &filename)
{
    size_t pos = filename.rfind('_');
    if (pos != std::string::npos)
    {
        size_t start = pos + 1;
        size_t end = filename.rfind('.');
        std::string labelStr = filename.substr(start, end - start);
        return std::stoi(labelStr);
    }
    return -1;
}

//void loadBinaryData(const std::string &filePath, fp_t **&data, unsigned int &outRows, unsigned int &outCols)
//{
//    const size_t timeSteps = TIME_STEPS;
//    const size_t flattenedSize = INPUT_SIZE;
//    std::ifstream file(filePath, std::ios::binary);
//    if (!file.is_open())
//    {
//        std::cerr << "Could not open the file " << filePath << std::endl;
//        return;
//    }
//
//    for (size_t t = 0; t < timeSteps; ++t)
//    {
//        std::vector<int16_t> buffer(flattenedSize);
//        file.read(reinterpret_cast<char *>(buffer.data()), flattenedSize * sizeof(int16_t));
//        for (size_t i = 0; i < flattenedSize; ++i)
//        {
//            data[t][i] = static_cast<fp_t>(buffer[i]);
//        }
//    }
//
//    outRows = timeSteps;
//    outCols = flattenedSize;
//}
//int main()
//{   
//    #ifdef LOAD
//    loadWeights();
//    loadInputs();
//    #endif
//
//
//    fp_t** inout = returnInputPtr<fp_t>(0);
//
//    fully_connected<double> FC1(0);
//    leaky<double> LEAKY1(0);
//    fully_connected<double> FC2(1);
//    leaky<double> LEAKY2(1);
//    fully_connected<double> FC3(2);
//    leaky<double> LEAKY3(2);
//
//    inout = FC1.run(inout);
//    inout = LEAKY1.run(inout);
//    
//    #ifdef TEST
//    std::ifstream f(path_to_inputs);
//    json data = json::parse(f);
//    for(unsigned int i = 0; i < TIME_STEPS; i++){
//        for(unsigned int j = 0; j < layer_size[1]; j++){
//            std::string json_idx = "lif" + std::to_string(1) + "_spikes";
//            fp_t true_val = data[json_idx][i][0][j];
//            if(abs(inout[i][j] - true_val) > PRECISION){
//                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
//                          << ", time step=" << i 
//                          << ", layer=" << 0
//                          << std::endl;
//                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
//            }
//        }
//    }          
//    #endif    
//
//    inout = FC2.run(inout);
//    inout = LEAKY2.run(inout);
//
//    #ifdef TEST
//    for(unsigned int i = 0; i < TIME_STEPS; i++){
//        for(unsigned int j = 0; j < layer_size[2]; j++){
//            std::string json_idx = "lif" + std::to_string(2) + "_spikes";
//            fp_t true_val = data[json_idx][i][0][j];
//            if(abs(inout[i][j] - true_val) > PRECISION){
//                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
//                          << ", time step=" << i 
//                          << ", layer=" << 1
//                          << std::endl;
//                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
//            }
//        }
//    }          
//    #endif
//
//    inout = FC3.run(inout);
//    inout = LEAKY3.run(inout);
//
//    #ifdef TEST
//    for(unsigned int i = 0; i < TIME_STEPS; i++){
//        for(unsigned int j = 0; j < layer_size[3]; j++){
//            std::string json_idx = "lif" + std::to_string(3) + "_spikes";
//            fp_t true_val = data[json_idx][i][0][j];
//            if(abs(inout[i][j] - true_val) > PRECISION){
//                std::cout << "Spike value mismatch in the output of a leaky layer at index=" << j 
//                          << ", time step=" << i 
//                          << ", layer=" << 2
//                          << std::endl;
//                std::cout << "Calculated value spike=" << inout[i][j] << ", while the true one is spike_true=" << true_val << std::endl;
//            }
//        }
//    }          
//    #endif
//    
//    cout << "Predicted class is: " << LEAKY3.predictClass() << endl;
//
//    for(unsigned int i = 0; i < TIME_STEPS; i++){
//        delete[] inout[i];
//    }
//    delete[] inout;
//    inout = nullptr;
//
//    return 0;
//}



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