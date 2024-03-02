#include "snn.hpp"

template <typename T> 
Leaky<T>::Leaky(unsigned int input_size, unsigned int output_size, float beta, float threshold, bool reset_type, float* U0){

    this->input_size = input_size;
    this->output_size = output_size;
    this->beta = beta;
    this->threshold = threshold;
    this->reset_type = reset_type;
    // Allocating memory for weights and membrane potential
    this->W.resize(output_size, input_size);
    this->U.resize(output_size, 1);

    // Membrane potential initialization
    for(unsigned int i = 0; i < output_size; i++)
        this->U(i) = U0[i];
}

template <typename T>
bool Leaky<T>::loadWeights(){
    W = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(this->output_size, this->input_size);
    return true;
}

template <typename T>
void Leaky<T>::writeInput(std::vector<char>& input){
    this->spikes.resize(this->input_size, 1);
    for(unsigned int i = 0; i < this->input_size; i++){
        this->spikes(i, 0) = input[i];
    }
}

template <typename T>
void Leaky<T>::readOutput(std::vector<char>& output){

    output.clear();
    for(unsigned int i = 0; i < this->output_size; i++){
        output.push_back((char)this->spikes(i, 0));
    }
}

template <typename T>
bool Leaky<T>::computeOutput(){

    if((this->spikes.rows() != this->input_size) || (this->spikes.cols() != 1) || (this->input_size == 0) || (this->output_size == 0))
        return false;
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> I = W * this->spikes.cast<T>();
    U = this->beta * U;
    U = U + I.template cast<float>();
    this->spikes.resize(this->output_size, 1);
    for(unsigned int i = 0; i < U.rows(); i++){
        this->spikes(i) = (U(i) > this->threshold) ? 1.f : 0.f;
    }

    if(this->reset_type){
        U = U - (threshold * this->spikes);
    }
    else{
        for(unsigned int i = 0; i < this->output_size; i++)
            if(this->spikes(i) == 1.f) 
                U(i) = 0;
    }

    return true;
}

template class Leaky<double>;
template class Leaky<float>;
template class Leaky<int>;
template class Leaky<char>;