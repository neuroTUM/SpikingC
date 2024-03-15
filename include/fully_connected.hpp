#ifndef FULLY_CONNECTED_HPP
#define FULLY_CONNECTED_HPP

#include "memory_access.hpp"
#include "linear_algebra.hpp"

template <typename T>
class fully_connected{
    public:
        fully_connected(unsigned int layer_num);
        ~fully_connected();
        T** run(T** input);
    private:
        T** input;
        T** output;
        weight_t** weight_ptr;
        weight_t* bias_ptr;
        unsigned int input_size;
        unsigned int output_size;
        unsigned int layer_num;
        void writeInput(T** input);
        void computeOutput();
        T** readOutput();
        void allocateMemory();
        void freeMemory();
};

template<typename T>
fully_connected<T>::fully_connected(unsigned int layer_num){
    this->input_size = layer_size[layer_num];
    this->output_size = layer_size[layer_num + 1];
    this->layer_num = layer_num;

    this->weight_ptr = returnWeightPtr<weight_t>(this->layer_num);
    this->bias_ptr = returnBiasPtr<weight_t>(this->layer_num);
}

template<typename T>
fully_connected<T>::~fully_connected(){
    this->input = nullptr;
    this->output = nullptr;
    delete[] this->weight_ptr;
    this->weight_ptr = nullptr;
    this->bias_ptr = nullptr;
}

template<typename T>
void fully_connected<T>::allocateMemory(){
    this->output = new T* [this->output_size];
    for(unsigned int i = 0; i < this->output_size; i++){
        this->output[i] = new T[TIME_STEPS];
    }
}

template<typename T>
void fully_connected<T>::freeMemory(){
    if(this->layer_num != 0){
        for(unsigned int i = 0; i < TIME_STEPS; i++){
            delete[] this->input[i];
        }
    }
    delete[] this->input;
    this->input = nullptr;
}

template<typename T>
void fully_connected<T>::writeInput(T** input){
    this->input = input;
}

template<typename T>
void fully_connected<T>::computeOutput(){

    matrixMAC(this->weight_ptr, this->output_size, this->input_size,
              this->input, this->input_size, TIME_STEPS,
              this->output, this->output_size, TIME_STEPS,
              this->bias_ptr, this->output_size);

    this->output = transposeMatrix<T>(this->output, this->output_size, TIME_STEPS);

    #ifdef TEST
    std::ifstream f(path_to_inputs);
    json data = json::parse(f);
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < layer_size[this->layer_num + 1]; j++){
            std::string json_idx = "fc" + std::to_string(this->layer_num + 1) + "_outputs";
            fp_t true_val = data[json_idx][i][0][j];
            if(abs(this->output[i][j] - true_val) > PRECISION){
                std::cout << "Output value mismatch in a fully connected layer at index=" << j 
                          << ", time step=" << i 
                          << ", layer=" << this->layer_num
                          << std::endl;
                std::cout << "Calculated value Isyn=" << this->output[i][j] << ", while the true one is fc_true=" << true_val << std::endl;
            }
        }
    }          
    #endif

}

template<typename T>
T** fully_connected<T>::readOutput(){
    return this->output;
}

template<typename T>
T** fully_connected<T>::run(T** input){
    this->allocateMemory();
    this->writeInput(input);
    this->computeOutput();
    T** ptr = this->readOutput();
    this->freeMemory();
    return ptr;    
}

#endif