#include <iostream>
#include "../include/snn.hpp"
#include "../include/Eigen/Dense"
#include <cstdlib>
#include <vector>
#include <fstream>
#include "../include/json.hpp"
#include <string>
#include <chrono>

using json = nlohmann::json;
using namespace std;

void loadInput(string file_path, vector<vector<float>>& input){
    std::ifstream f(file_path);
    json data = json::parse(f);
    vector<float> input_col;

    for(unsigned int row = 0; row < 2312; row++){
        for(unsigned int col = 0; col < 31; col++){
            input_col.push_back(data["inputs"][col][0][row]);
        }
        input.push_back(input_col);
        input_col.clear();
    }
}

int main()
{
    string weights_path = "../../models/SNN_3L_simple_LIF_NMNIST/extra_formats/model_weights.json";
    string inputs_path = "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json";

    unsigned int num_layers = 3;
    unsigned int time_steps = 31;
    vector<unsigned int> layer_size;
    layer_size.insert(layer_size.end(), {2312, 2312 / 4, 2312 / 8, 10});
    vector<float> beta;
    beta.insert(beta.end(), {0.5, 0.5, 0.5});
    vector<float> threshold;
    threshold.insert(threshold.end(), {2.5, 8.0, 4.0});
    vector<bool> reset_type;
    reset_type.insert(reset_type.end(), {true, true, true});

    vector<vector<float>> input;
    loadInput(inputs_path, input);

    cout << "Inputs loaded from the JSON file" << endl;

    Leaky<double> model(num_layers, time_steps, layer_size, weights_path, beta, threshold, reset_type);
    model.loadInput(input);

    cout << "Inputs passed to the model" << endl;

    auto start = chrono::high_resolution_clock::now();
    model.computeOutputSpikes();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double total_time = duration.count();

    cout << "Output spikes computed" << endl;

    cout << model.predictClass() << endl;

    cout << "Prediction generated" << endl;

    /* Performance measurments */
    cout << "Processor load for loading weights: " << (model.load_weights_time / total_time) * 100.0f << endl;
    cout << "Processor load for fully connected layers: " << (model.compute_fc_time / total_time) * 100.0f << endl;
    cout << "Processor load for LIF neurons: " << (model.compute_leaky_time / total_time) * 100.0f << endl;

    return 0;
}