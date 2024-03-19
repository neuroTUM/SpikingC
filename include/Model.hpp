#ifndef MODEL_HPP
#define MODEL_HPP

#include "LIF.hpp"
#include "Linear.hpp"

class Model{
    public:
        Model();
        ~Model();
        void run(cfloat_array_t* In);
        void resetState();
        unsigned int predict();
    private:
        LIF** LIF_layers;
        Linear** Linear_layers;
        cfloat_array_t floatOut;
        spike_array_t spikeOut;
        unsigned int *actPred;
        unsigned int lif_num;
        unsigned int linear_num;
};

Model::Model(){

    this->actPred = new unsigned int [layer_size[NUM_LAYERS]];

    this->LIF_layers = new LIF*[NUM_LAYERS / 2];
    this->Linear_layers = new Linear*[NUM_LAYERS / 2];

    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(layer_type[i] == "LIF"){
            this->LIF_layers[i / 2] = new LIF(i);
        }
        else if(layer_type[i] == "Linear"){
            this->Linear_layers[i / 2] = new Linear(i);
        }        
    }
}

Model::~Model(){    
    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(layer_type[i] == "LIF"){
            delete this->LIF_layers[i / 2];
        }
        else if(layer_type[i] == "Linear"){
            delete this->Linear_layers[i / 2];
        }        
    }
    delete[] this->LIF_layers;
    delete[] this->Linear_layers;
    delete[] this->actPred;
}

void Model::resetState(){
    unsigned long long idx = 0;
    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(layer_type[i] == "LIF")
            idx += layer_size[i + 1];
    }

    for(unsigned int i = 0; i < idx; i++){
        mem_potential[i] = 0;
    }

    for(unsigned int i = 0; i < (idx / 8) + (NUM_LAYERS / 2); i++){
        spike_memory[i] = 0;
    }

    for(unsigned int i = 0; i < layer_size[NUM_LAYERS]; i++){
        this->actPred[i] = 0;
    }
}

void Model::run(cfloat_array_t* In){

    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(layer_type[i] == "LIF"){
            this->spikeOut.size = layer_size[i + 1];
            this->spikeOut.ptr  = returnSpikePtr(i);
            this->LIF_layers[i / 2][0].computeOutput(&this->floatOut, &this->spikeOut);
        }
        else if(layer_type[i] == "Linear"){
            this->floatOut.size = layer_size[i + 1];
            if(i == 0){
                this->floatOut.ptr  = &scrachpad_memory[INPUT_SIZE];           
                this->Linear_layers[i / 2][0].computeOutput(In, &this->floatOut);
            }
            else{
                this->floatOut.ptr  = scrachpad_memory;            
                this->Linear_layers[i / 2][0].computeOutput(&this->spikeOut, &this->floatOut);
            }
        }        
    }

    this->spikeOut.ptr = returnSpikePtr(NUM_LAYERS - 1);
    unsigned int idx = 0;
    unsigned char cnt;
    for(unsigned int i = 0; i < layer_size[NUM_LAYERS]; i += (sizeof(spike_t) * 8)){
        spike_t val = this->spikeOut.ptr[idx++];
        cnt = 0;
        for(unsigned int k = i; k < min(i + (sizeof(spike_t) * 8), layer_size[NUM_LAYERS]); k++){
            this->actPred[k] += (unsigned int)((val >> cnt++) & MASK);
        }
    }
}

unsigned int Model::predict(){
    unsigned int max = this->actPred[0];
    unsigned int idx = 0;
    for(unsigned int i = 1; i < layer_size[NUM_LAYERS]; i++){
        if(max < this->actPred[i]){
            max = this->actPred[i];
            idx = i;
        }
    }

    return idx;    
}

#endif