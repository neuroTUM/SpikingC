#include "../include/LIF.h"

#ifdef TEST
void testLIF(lif_t* layer, const spike_array_t* spikes){
    
    int rows, cols;
    char filename[256];

    /**************************************** Membrane potential ****************************************/
    sprintf(filename, "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/mem%u/mem%u_timestep_%u.csv", 
                      (layer->layer_num / 2) + 1, (layer->layer_num / 2) + 1, layer->curr_time_step);
    
    float **data = readBinary(filename, &rows, &cols);
    if (!data || rows < 1)
    {
        fprintf(stderr, "Failed to load membrane potentials for timestep %u and LIF layer %u\n", layer->curr_time_step, (layer->layer_num / 2) + 1);
        exit(1);
    }

    for (unsigned int i = 0; i < (unsigned int)cols; i++)
    {
        if(fabs(layer->U.ptr[i] - data[0][i]) > PRECISION){
            fprintf(stderr, "Layer_%u (LIF): Membrane potential value mismatch at position %u and time step %u.\nCorrect value = %f\tCalculated value = %f\n", 
                             layer->layer_num, i, layer->curr_time_step, data[0][i], layer->U.ptr[i]);
            exit(1);            
        }
    }
    freeCSVData(data, rows);

    /********************************************* Spikes *********************************************/
    
    sprintf(filename, "../../models/SNN_3L_simple_LIF_NMNIST/intermediate_outputs/lif%u/lif%u_spikes_timestep_%u.csv", 
                      (layer->layer_num / 2) + 1, (layer->layer_num / 2) + 1, layer->curr_time_step);

    data = readBinary(filename, &rows, &cols);
    if (!data || rows < 1)
    {
        fprintf(stderr, "Failed to load spikes for timestep %u and LIF layer %u\n", layer->curr_time_step, (layer->layer_num / 2) + 1);
        exit(1);
    }

    spike_t single_spike;
    for(unsigned int i = 0; i < (unsigned int)cols; i += sizeof(spike_t) * 8){
        single_spike = 0;
        spike_t val = spikes->ptr[i / (sizeof(spike_t) * 8)];        
        for(unsigned int j = i; j < i + sizeof(spike_t) * 8 && j < (unsigned int)cols; j++){
            single_spike = BITVALUE(val, j - i);
            if(fabs((float)single_spike - data[0][j]) > PRECISION){
                fprintf(stderr, "Layer_%u (LIF): Spike value mismatch at position %u and time step %u.\nCorrect value = %f\tCalculated value = %f\n", 
                                layer->layer_num, j, layer->curr_time_step, data[0][i], (float)single_spike);
                exit(1);            
            }            
        }
    }
    freeBinaryData(data, rows);

    layer->curr_time_step++;
    if(layer->curr_time_step >= TIME_STEPS)
        layer->curr_time_step = 0;    
}
#endif

void clearLIF(lif_t* layer){
    layer->U.ptr = NULL;
}

void computeOutput(lif_t* layer, cfloat_array_t* In, spike_array_t* Out){
    
    /*  
        IMPORTANT:
        Only the 'subtract' reset type is supported!

        Observations:
        1. Spikes are generated with the updated values of potentials at the same
           time step
        2. I have to save the spike values from the previous time step
        3. S[t] = (U[t] > Threshold) ? 1 : 0
           U[t] = beta * (U[t-1] - S[t-1] * Threshold) + Isyn[t]
    */

    spike_t in, out;
    for(unsigned int i = 0; i < layer->U.size; i += sizeof(spike_t) * 8){
        out = 0;
        spike_t val = Out->ptr[i / (sizeof(spike_t) * 8)];        
        for(unsigned int j = i; j < i + sizeof(spike_t) * 8 && j < layer->U.size; j++){
            if(reset_type[layer->layer_num]){
                in = BITVALUE(val, j - i);
                layer->U.ptr[j] = Beta[layer->layer_num] * (layer->U.ptr[j] - in * threshold[layer->layer_num]) + In->ptr[j];
                in = (layer->U.ptr[j] > threshold[layer->layer_num]) ? 0x1 : 0x0;
                out |= (in << (j - i));
            }
            else{
                /* Zero reset not supported */
            }
        }
        Out->ptr[i / (sizeof(spike_t) * 8)] = out;
    }

    #ifdef TEST
    testLIF(layer, Out);
    #endif
}

void initLIF(lif_t* layer, unsigned int layer_num){
    layer->layer_num            = layer_num;
    layer->U.ptr                = returnMemPotentialPtr(layer_num);
    layer->U.size               = layer_size[layer_num];
    layer->clearLIF_fptr        = &clearLIF;
    layer->computeOutput_fptr   = &computeOutput;
    
    #ifdef TEST
    layer->curr_time_step       = 0;
    #endif
}