#include "../include/snn.hpp"	

template <typename T> 
Leaky<T>::Leaky(unsigned int num_layers, unsigned int time_steps, vector<unsigned int> layer_size, 
                string path, vector<float> beta, vector<float> threshold, vector<bool> reset_type){

    this->num_layers    = num_layers;
    this->time_steps    = time_steps;
    this->layer_size    = layer_size;
    this->beta          = beta;
    this->threshold     = threshold;
    this->reset_type    = reset_type;
    this->curr_layer    = 0;
    this->path          = path;

    this->load_weights_time = 0;
    this->compute_fc_time = 0;
    this->compute_leaky_time = 0;

    // Initializing U to zero for all layers
    auto vector = new Eigen::Matrix<float, Eigen::Dynamic, 1>;
    for(unsigned int i = 0; i < this->num_layers; i++){
        *vector = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(this->layer_size[i + 1], 1);
        this->U.push_back(*vector);
    }

    vector->resize(0, 1);
    delete vector;
}

template <typename T>
void Leaky<T>::loadInput(vector<vector<float>>& input){
    spikes.resize(this->layer_size[this->curr_layer], this->time_steps);
    for(unsigned int row = 0; row < this->spikes.rows(); row++){
        for(unsigned int col = 0; col < this->spikes.cols(); col++){
            spikes(row, col) = input[row][col];
        }
    }
}

template <typename T>
bool Leaky<T>::loadWeights(){

    /* Weights and biases are only loaded for the current layer because in case of 
       larger networks this could explode in terms of memory requirements
    */

    std::ifstream f(path);
    json data = json::parse(f);
    this->W.resize(this->layer_size[this->curr_layer + 1], this->layer_size[this->curr_layer]);
    this->B.resize(this->layer_size[this->curr_layer + 1], this->time_steps);

    string weights = "fc" + to_string(this->curr_layer + 1) + ".weight";
    string biases = "fc" + to_string(this->curr_layer + 1) + ".bias";

    for(unsigned int row = 0; row < this->W.rows(); row++){
        for(unsigned int col = 0; col < this->W.cols(); col++){
            unsigned int idx = (row * W.cols()) + col;
            this->W(row, col) = data[weights][idx];
        }
    }

    for(unsigned int col = 0; col < this->B.cols(); col++){
        for(unsigned int row = 0; row < this->B.rows(); row++){
            this->B(row, col) = data[biases][row];
        }
    }

    return true;
}

template <typename T>
void Leaky<T>::performFC(){
    auto M = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    *M = (this->W * this->spikes.cast<T>()) + this->B;
    this->Isyn = M->template cast<float>();
    delete M;

    #ifdef TEST_FC
        float fc_true;
        string fc_layer = "fc" + to_string(this->curr_layer + 1) + "_outputs";
        std::ifstream f("../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json");
        json data = json::parse(f);
        for(unsigned int col = 0; col < this->Isyn.cols(); col++){
            for(unsigned int row = 0; row < this->Isyn.rows(); row++){
                fc_true = data[fc_layer][col][0][row];
                if(abs(this->Isyn(row, col) - fc_true) > PRECISION){
                    cout << "Output value mismatch of a fully connected layer at index=" << row << ", time step=" << col  << ", layer=" << this->curr_layer + 1 << endl;
                    cout << "Calculated value Isyn=" << this->Isyn(row, col) << ", while the true one is fc_true=" << fc_true << endl;
                }
            }
        }
    #endif
}

template <typename T>
void Leaky<T>::performLeaky(){

    #ifdef TEST_SPIKES
        float spikes_true;
        std::ifstream f("../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json");
        json data = json::parse(f);
    #endif

    #ifdef TEST_MEM
        float U_true;
        std::ifstream f("../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/layer_outputs.json");
        json data = json::parse(f);
    #endif
    

    this->spikes.resize(this->layer_size[this->curr_layer + 1], this->time_steps);
        
    for(unsigned int i = 0; i < this->time_steps; i++){
        for(unsigned int j = 0; j < this->spikes.rows(); j++){
            this->spikes(j, i) = (this->U[this->curr_layer](j) < this->threshold[this->curr_layer]) ? 0.f : 1.f;
            this->U[this->curr_layer](j) = this->beta[this->curr_layer] * this->U[this->curr_layer](j);
            this->U[this->curr_layer](j) = this->U[this->curr_layer](j) + this->Isyn(j, i);
            if(this->reset_type[this->curr_layer]){
                this->U[this->curr_layer](j) = this->U[this->curr_layer](j) - (this->threshold[this->curr_layer] * this->spikes(j, i));
            }
            else{
                this->U[this->curr_layer](j) = (this->spikes(j, i) > 0.f) ? 0.f : this->U[this->curr_layer](j);
            }

            #ifdef TEST_MEM
                string mem_layer = "mem" + to_string(this->curr_layer + 1);
                U_true = data[mem_layer][i][0][j];
                if(abs(this->U[this->curr_layer](j) - U_true) > PRECISION && (this->curr_layer == 0) && (i == 2)){
                    cout << "Membrane potential's value mismatch at index=" << j << ", time step=" << i << ", layer=" << this->curr_layer + 1 << endl;
                    cout << "Calculated value of U=" << this->U[this->curr_layer](j) << ", while the true one is U_true=" << U_true << endl;
                }
            #endif

            #ifdef TEST_SPIKES
                string lif_layer = "lif" + to_string(this->curr_layer + 1) + "_spikes";
                spikes_true = data[lif_layer][i][0][j];
                if(abs(this->spikes(j, i) - spikes_true) > PRECISION && (this->curr_layer == 0) && (i == 0)){
                    cout << "Spike value mismatch at index=" << j << ", time step=" << i << ", layer=" << this->curr_layer + 1 << endl;
                    cout << "Calculated value of spikes=" << this->spikes(j, i) << ", while the true one is spikes_true=" << spikes_true << endl;
                }
            #endif
        }
    }
}

template <typename T>
void Leaky<T>::computeOutputSpikes(){

    for(unsigned int i = 0; i < this->num_layers; i++){
        #ifdef MEASURE_PERFORMANCE
            auto start = chrono::high_resolution_clock::now();
        #endif
        this->loadWeights();
        #ifdef MEASURE_PERFORMANCE
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> duration = end - start;
            this->load_weights_time += duration.count();
            start = chrono::high_resolution_clock::now(); 
        #endif
        this->performFC();
        #ifdef MEASURE_PERFORMANCE
            end = chrono::high_resolution_clock::now();
            duration = end - start;
            this->compute_fc_time += duration.count();
            start = chrono::high_resolution_clock::now();         
        #endif
        this->performLeaky();
        #ifdef MEASURE_PERFORMANCE
            end = chrono::high_resolution_clock::now();
            duration = end - start;
            this->compute_leaky_time += duration.count();
        #endif
        this->curr_layer++;
    }

    // cout << this->spikes << endl;
    this->curr_layer = 0;
}

template <typename T>
void Leaky<T>:: initMembranePotential(){
    for(unsigned int i = 0; i < this->num_layers; i++){
        this->U[i] = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(this->layer_size[i + 1], 1);
    }
}

template <typename T>
unsigned int Leaky<T>::predictClass(){

    unsigned int pred_class = 0;
    auto curr_max = this->spikes.row(0).sum();
    for(unsigned int i = 1; i < this->spikes.rows(); i++){
        if(this->spikes.row(i).sum() > curr_max){
            curr_max = this->spikes.row(i).sum();
            pred_class = i;
        }
    }

    return pred_class;
}

template class Leaky<double>;
template class Leaky<float>;
template class Leaky<int>;
// template class Leaky<char>;