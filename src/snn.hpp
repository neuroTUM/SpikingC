#ifndef SNN_HPP
#define SNN_HPP

#include "../include/Eigen/Dense"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "../include/json.hpp"

using json = nlohmann::json;
using namespace std;

template <typename T> 
class Leaky{

    private:

        /* Class parameters */

        // Current layer being processed
        unsigned int curr_layer;

        // Number of layers
        unsigned int num_layers;

        // Batch size
        unsigned int batch_size;

        // Number of time steps
        unsigned int time_steps;

        // Layer dimensions
        /* input_size = [4, 8, 4, 8, 2]
           L1: IN = 4, OUT = 8
           L2: IN = 8, OUT = 4
           L3: IN = 4, OUT = 8
           L4: IN = 8, OUT = 2
        */
        vector<unsigned int> layer_size;

        // Membrane potential decay rate
        vector<float> beta;

        // Membrane threshold
        vector<float> threshold;

        // 1 = subtract mechanism, 0 = zero mechanisam
        vector<bool> reset_type;

        // Path to weights and biases
        string path;

        // Weights
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;

        // Biases
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B;

        // Membrane potential
        vector<Eigen::Matrix<float, Eigen::Dynamic, 1>> U;

        // Synaptic current
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Isyn;

        // Input and output spikes
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> spikes;

        /* Member Functions */

        // It should load weights in the weight matrix
        // So far it generates random values
        bool loadWeights();

        // Performs matrix multiply and accumulate for a fully connected layer
        void performFC();

        // Performs necessary computations for a LIF neuron
        void performLeaky();

        // Performs the whole sequence of operations for a single batch
        void computeBatch();
    
    public:

        /* Member Functions */

        // Constructor
        Leaky(unsigned int num_layers, unsigned int batch_size, unsigned int time_steps, vector<unsigned int> layer_size, 
              string path, vector<float> beta, vector<float> threshold, vector<bool> reset_type);

        // Loads input spikes
        void loadInput(vector<vector<float>>& input);

        // Reads output spikes
        void readOutput(vector<char>& input);

};

#endif