#ifndef LEAKY_HPP
#define LEAKY_HPP

#include "network_config.hpp"

template<typename T>
class leaky{
    public:
        leaky(unsigned int layer_num);
        ~leaky();
        unsigned int predictClass();
        T** run(T** input);
    private:
        T** input;
        T** output;
        T* U;
        unsigned int size;
        unsigned int layer_num;
        void allocateMemory();
        void freeMemory();
        void initLeaky();
        void writeInput(T** input);
        void computeOutput();
        T** readOutput();

};

template<typename T>
leaky<T>::leaky(unsigned int layer_num){

    this->size = layer_size[layer_num + 1];
    this->layer_num = layer_num;

    this->input = nullptr;
}

template<typename T>
leaky<T>::~leaky(){
    this->U = nullptr;
    this->output = nullptr;
}

template<typename T>
void leaky<T>::allocateMemory(){

    this->U = new T [this->size];

    this->output = new T* [TIME_STEPS];
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        this->output[i] = new T[this->size];
    }
}

template<typename T>
void leaky<T>::freeMemory(){
    delete[] this->U;
    this->U = nullptr;
}

template<typename T>
void leaky<T>::initLeaky(){
    for(unsigned int i = 0; i < this->size; i++){
        this->U[i] = 0;
    }
}

template<typename T>
void leaky<T>::writeInput(T** input){
    this->input = input;
}

template<typename T>
void leaky<T>::computeOutput(){

    /*  IMPORTANT:
        Only the 'subtract' reset type is supported!
    */

    /* Observations:
        1. Spikes are generated with the updated values of potentials at the same
           time step
        2. I have to save the spike values from the previous time step
        3. S[t] = (U[t] > Threshold) ? 1 : 0
           U[t] = beta * (U[t-1] - S[t-1] * Threshold) + Isyn[t]
    */
    #ifdef TEST
    std::ifstream f(path_to_inputs);
    json data = json::parse(f);
    #endif


    /* The input and output matrices are stored in column major order */
    T prev_spike;
    for(unsigned int i = 0; i < TIME_STEPS; i++){
        for(unsigned int j = 0; j < this->size; j++){
            if(reset_type[this->layer_num]){
                prev_spike = (i == 0) ? 0 : this->output[i - 1][j];
                this->U[j] = beta[this->layer_num] * (this->U[j] - prev_spike * threshold[this->layer_num]) + this->input[i][j];
                this->output[i][j] = (this->U[j] > threshold[this->layer_num]) ? 1 : 0;
            }
            else{
                if(this->output[i][j] == 0){
                    this->U[j] = 0;
                }
            }

            #ifdef TEST
            std::string json_idx = "mem" + std::to_string(this->layer_num + 1);
            fp_t true_val = data[json_idx][i][0][j];
            if(abs(this->U[j] - true_val) > PRECISION){
                std::cout << "Membrane potential mismatch in a leaky layer at index=" << j 
                        << ", time step=" << i 
                        << ", layer=" << this->layer_num
                        << std::endl;
                std::cout << "Calculated value U=" << this->U[j] << ", while the true one is U_true=" << true_val << std::endl;
            }     
            #endif

        }

    }

    for(unsigned int i = 0; i < TIME_STEPS; i++){
        delete[] this->input[i];
    }
    delete[] this->input;
    this->input = nullptr;
}

template<typename T>
T** leaky<T>::readOutput(){
    return this->output;
}

template<typename T>
unsigned int leaky<T>::predictClass(){
    int prediction = -1;
    T max;
    T temp;
    for(unsigned int i = 0; i < this->size; i++){
        temp = 0;
        for(unsigned int j = 0; j < TIME_STEPS; j++){
            temp += this->output[j][i];
        }

        if(prediction < 0){
            max = temp;
            prediction = i;
        }
        else{
            if(temp > max){
                max = temp;
                prediction = i;
            }
        }
    }

    return prediction;
}

template<typename T>
T** leaky<T>::run(T** input){
    this->allocateMemory();
    this->initLeaky();
    this->writeInput(input);
    this->computeOutput();
    T** ptr = this->readOutput();
    this->freeMemory();
    return ptr;
}

#endif