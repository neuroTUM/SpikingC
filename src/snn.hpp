#ifndef SNN_HPP
#define SNN_HPP

#include <../Eigen/Dense>
#include <vector>
#include <iostream>

template <typename T> class Leaky{

    private:

         /* Class parameters */

        // Number of input dimensions
        unsigned int input_size;

        // Number of neurons in this layer
        unsigned int output_size; 

        // Membrane potential decay rate
        float beta;

        // Membrane threshold
        float threshold;

        // 1 = subtract mechanism, 0 = zero mechanisam
        bool reset_type;

        // Weights
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;

        // Membrane potential
        Eigen::Matrix<float, Eigen::Dynamic, 1> U;

        // Input and output spikes
        Eigen::Matrix<float, Eigen::Dynamic, 1> spikes;
    
    public:

        /* Member Functions */

        // Constructor
        Leaky(unsigned int input_size, unsigned int output_size, float beta, float threshold, bool reset_type, float* U0);

        // It should load weights in the weight matrix
        // So far it generates random values
        bool loadWeights();

        // Performes necessary computations
        bool computeOutput();

        // Loads input spikes
        void writeInput(std::vector<char>& input);

        // Reads output spikes
        void readOutput(std::vector<char>& input);

};

#endif