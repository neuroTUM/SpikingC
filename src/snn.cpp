#include "snn.hpp"

template <typename T> 
Leaky<T>::Leaky(unsigned int num_layers, unsigned int batch_size, unsigned int time_steps, vector<unsigned int> layer_size, 
                vector<float> beta, vector<float> threshold, vector<bool> reset_type){

    this->num_layers    = num_layers;
    this->batch_size    = batch_size;
    this->time_steps    = time_steps;
    this->layer_size    = layer_size;
    this->beta          = beta;
    this->threshold     = threshold;
    this->reset_type    = reset_type;
    this->curr_layer    = 0;


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
    spikes.resize(this->layer_size[this->curr_layer], this->batch_size);
    for(unsigned int row = 0; row < this->spikes.rows(); row++){
        for(unsigned int col = 0; col < this->spikes.cols(); col++){
            spikes(row, col) = input[row][col];
        }
    }
}

template <typename T>
bool Leaky<T>::loadWeights(){
    // Weights and biases are only loaded for the current layer because in case of larger networks this could explode in terms of memory requirements
    this->W = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(this->layer_size[this->curr_layer + 1], this->layer_size[this->curr_layer]);
    this->B = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(this->layer_size[this->curr_layer + 1], this->batch_size);
    return true;
}

template <typename T>
void Leaky<T>::performFC(){
    auto M = new Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    *M = (this->W * this->spikes.cast<T>()) + this->B;
    this->Isyn = M->template cast<float>();
    delete M;
}

template <typename T>
void Leaky<T>::performLeaky(){
    
    this->U[this->curr_layer] = this->beta[this->curr_layer] * this->U[this->curr_layer];
    this->spikes.resize(this->layer_size[this->curr_layer + 1], this->batch_size);
    
    for(unsigned int i = 0; i < this->spikes.cols(); i++){
        this->U[this->curr_layer] = this->U[this->curr_layer] + this->Isyn.col(i);
        for(unsigned int j = 0; j < this->spikes.rows(); j++){
            this->spikes(j, i) = (this->U[this->curr_layer](j) < this->threshold[this->curr_layer]) ? 0.f : 1.f;
            if(this->reset_type[this->curr_layer]){
                this->U[this->curr_layer](j) = this->U[this->curr_layer](j) - (this->threshold[this->curr_layer] * this->spikes(j, i));
            }
            else{
                this->U[this->curr_layer](j) = (this->spikes(j, i) > 0.f) ? 0.f : this->U[this->curr_layer](j);
            }
        }
    }

    this->curr_layer++;
}

template <typename T>
void Leaky<T>::computeBatch(){

    this->spikes = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(this->layer_size[0], this->batch_size);

    for(unsigned int i = 0; i < this->num_layers; i++){
        this->loadWeights();
        this->performFC();
        this->performLeaky();
    }

    cout << this->spikes << endl;

    this->curr_layer = 0;
}

template <typename T>
void Leaky<T>::readOutput(vector<char>& output){
}

template class Leaky<double>;
template class Leaky<float>;
template class Leaky<int>;
template class Leaky<char>;