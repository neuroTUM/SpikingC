#include <iostream>
#include "snn.hpp"
#include "../include/Eigen/Dense"
#include <cstdlib>
#include <vector>

using namespace std;

int main()
{
    srand((unsigned) time(NULL));

    unsigned int num_layers = 3;
    unsigned int batch_size = 128;
    unsigned int time_steps = 31;
    vector<unsigned int> layer_size;
    layer_size.insert(layer_size.end(), {2312, 2312 / 4, 2312 / 8, 10});
    vector<float> beta;
    beta.insert(beta.end(), {0.5, 0.5, 0.5});
    vector<float> threshold;
    threshold.insert(threshold.end(), {2.5, 8.0, 4.0});
    vector<bool> reset_type;
    reset_type.insert(reset_type.end(), {false, false, true, false});
    Leaky<char> model(num_layers, batch_size, time_steps, layer_size, beta, threshold, reset_type);

    return 0;
}