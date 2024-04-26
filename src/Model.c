#include "../include/Model.h"

void clearModel(model_t *model)
{
    if (model != NULL)
    {
        // Iterate through each layer
        for (unsigned int i = 0; i < NUM_LAYERS; i++)
        {
            // Clear Linear layers
            if (strcmp(layer_type[i], "Linear") == 0 && model->layers[i].linear_ptr != NULL)
            {
                if (model->layers[i].linear_ptr->clearLinear_fptr != NULL)
                {
                    model->layers[i].linear_ptr->clearLinear_fptr(model->layers[i].linear_ptr);
                }
                free(model->layers[i].linear_ptr);
                model->layers[i].linear_ptr = NULL; // Prevent dangling pointer
            }
            // Clear LIF layers
            else if (strcmp(layer_type[i], "LIF") == 0 && model->layers[i].lif_ptr != NULL)
            {
                if (model->layers[i].lif_ptr->clearLIF_fptr != NULL)
                {
                    model->layers[i].lif_ptr->clearLIF_fptr(model->layers[i].lif_ptr);
                }
                free(model->layers[i].lif_ptr);
                model->layers[i].lif_ptr = NULL; // Prevent dangling pointer
            }
        }
        // Free the array of layers
        free(model->layers);
        model->layers = NULL; // Prevent dangling pointer

        // Free the actPred array
        free(model->actPred);
        model->actPred = NULL; // Prevent dangling pointer
    }
}

void resetState(model_t* model){
    unsigned int offset = getOffset(NUM_LAYERS, 'V', "LIF");

    for(unsigned int i = 0; i < offset; i++){
        mem_potential[i] = 0;
    }

    unsigned int num_lif_layers = 0;
    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(strcmp(layer_type[i], "LIF") == 0)
            num_lif_layers++;
    }

    for(unsigned int i = 0; i < (offset / 8) + (num_lif_layers / 2); i++){
        spike_memory[i] = 0;
    }

    for(unsigned int i = 0; i < layer_size[NUM_LAYERS]; i++){
        model->actPred[i] = 0;
    }
}

void run(model_t* model, cfloat_array_t* In){
    
    /* Forward path */
    for(unsigned int i = 0; i < NUM_LAYERS; i++){
        if(strcmp(layer_type[i], "Linear") == 0){
            model->floatOut.size = layer_size[i + 1];
            if(i == 0){
                model->floatOut.ptr  = &scrachpad_memory[INPUT_SIZE];           
                model->layers[i].linear_ptr->matrixVectorMul_fptr(&(model->layers[i].linear_ptr->W), 
                                                                  &(model->layers[i].linear_ptr->B),
                                                                  In,
                                                                  &(model->floatOut));
            }
            else{
                model->floatOut.ptr  = scrachpad_memory;            
                model->layers[i].linear_ptr->matrixVectorMulSparse_fptr(&(model->layers[i].linear_ptr->W), 
                                                                    	&(model->layers[i].linear_ptr->B),
                                                                        &(model->spikeOut),
                                                                        &(model->floatOut));
            }
        }
        else if(strcmp(layer_type[i], "LIF") == 0){
            model->spikeOut.size = layer_size[i + 1];
            model->spikeOut.ptr  = returnSpikePtr(i);
            model->layers[i].lif_ptr->computeOutput_fptr(model->layers[i].lif_ptr, &(model->floatOut), &(model->spikeOut));
        }
    }

    /* Updated the active prediction array after each time step*/
    for(unsigned int i = 0; i < layer_size[NUM_LAYERS]; i += (sizeof(spike_t) * 8)){
        spike_t val = model->spikeOut.ptr[i / (sizeof(spike_t) * 8)];
        for(unsigned int k = i; k < i + (sizeof(spike_t) * 8) && k < layer_size[NUM_LAYERS]; k++){
            model->actPred[k] += BITVALUE(val, (k-i));
        }
    }
}

unsigned int predict(model_t* model){
    unsigned int max = model->actPred[0];
    unsigned int idx = 0;
    for(unsigned int i = 1; i < layer_size[NUM_LAYERS]; i++){
        if(max < model->actPred[i]){
            max = model->actPred[i];
            idx = i;
        }
    }

    return idx;    
}

void initModel(model_t *model)
{
    model->clearModel_fptr = &clearModel;
    model->resetState_fptr = &resetState;
    model->run_fptr = &run;
    model->predict_fptr = &predict;
    model->layers = (layer_instance_t *)malloc(sizeof(layer_instance_t) * NUM_LAYERS);
    model->actPred = (unsigned int *)malloc(sizeof(unsigned int) * layer_size[NUM_LAYERS]);

    for (unsigned int i = 0; i < NUM_LAYERS; i++)
    {
        if (strcmp(layer_type[i], "Linear") == 0)
        {
            model->layers[i].linear_ptr = (linear_t *)malloc(sizeof(linear_t));
            initLinear(model->layers[i].linear_ptr, i);
        }
        else if (strcmp(layer_type[i], "LIF") == 0)
        {
            model->layers[i].lif_ptr = (lif_t *)malloc(sizeof(lif_t)); // Corrected type casting
            initLIF(model->layers[i].lif_ptr, i);
        }
    }
}
