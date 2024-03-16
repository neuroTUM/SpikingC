#include "../include/fully_connected.hpp"
#include "../include/leaky.hpp"
#include "../include/network_config.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

fp_t **loadBinaryData(const std::string &filePath, unsigned int &outRows, unsigned int &outCols)
{
    const size_t timeSteps = 30;              // Assuming 30 timesteps
    const size_t flattenedSize = 2 * 34 * 34; // Flatten the spatial dimensions
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file " << filePath << std::endl;
        return nullptr;
    }

    fp_t **data = new fp_t *[timeSteps];
    for (size_t t = 0; t < timeSteps; ++t)
    {
        data[t] = new fp_t[flattenedSize];
        std::vector<int16_t> buffer(flattenedSize);
        file.read(reinterpret_cast<char *>(buffer.data()), flattenedSize * sizeof(int16_t));
        for (size_t i = 0; i < flattenedSize; ++i)
        {
            data[t][i] = static_cast<fp_t>(buffer[i]);
        }
    }

    outRows = timeSteps;
    outCols = flattenedSize;
    return data;
}

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

#ifdef LOAD
    loadWeights();
    loadInputs();
#endif

    const std::string directoryPath = "/home/copparihollmann/neuroTUM/SpikingCpp/tests/NMNIST";
    unsigned int totalPredictions = 0;
    unsigned int correctPredictions = 0;

    // Determine the total number of .bin files
    unsigned int totalFiles = 0;
    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            totalFiles++;
        }
    }

    // Current file being processed
    unsigned int currentFileNumber = 0;

    for (const auto &entry : fs::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
        {
            currentFileNumber++;
            std::string filePath = entry.path().string();
            int trueLabel = extractLabelFromFilename(entry.path().filename().string());
            unsigned int rows, cols;
            fp_t **Data = loadBinaryData(filePath, rows, cols);
            if (!Data)
                continue;

            fully_connected<double> FC1(0);
            leaky<double> LEAKY1(0);
            fully_connected<double> FC2(1);
            leaky<double> LEAKY2(1);
            fully_connected<double> FC3(2);
            leaky<double> LEAKY3(2);

            Data = FC1.run(Data);
            Data = LEAKY1.run(Data);
            Data = FC2.run(Data);
            Data = LEAKY2.run(Data);
            Data = FC3.run(Data);
            Data = LEAKY3.run(Data);

            int predictedLabel = LEAKY3.predictClass();

            

            //for (unsigned int i = 0; i < TIME_STEPS; i++)
            //{
            //    delete[] Data[i];
            //}
            //delete[] Data;
            //Data = nullptr;


            // Assume getPredictedLabel() is a method that returns the predicted label
            // unsigned int predictedLabel = getPredictedLabel(inputData);

            if (predictedLabel == trueLabel) correctPredictions++;
            totalPredictions++;

            cout << "Predicted class is: " << predictedLabel << " // LABEL: " << trueLabel << endl;
            std::cout << "Processing input " << currentFileNumber << " out of " << totalFiles << std::endl;
        }
        // print how many files are left to process
        // Print the current progress
        
    }

    cout << "WE GOT UP TO THIS POINT" << endl;

    if (totalPredictions > 0)
    {
        cout << "ALMOST THERE" << endl;
        float accuracy = static_cast<float>(correctPredictions) / totalPredictions;
        std::cout << "Accuracy: " << accuracy << std::endl;
    }
    else
    {
        std::cout << "No files processed." << std::endl;
    }

    return 0;
}