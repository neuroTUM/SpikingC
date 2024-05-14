#include "../include/SNNconfig.h"

unsigned int    layer_size[NUM_LAYERS + 1]          = {INPUT_SIZE, L1_SIZE_OUT, LIF1_SIZE, 
                                                       L2_SIZE_OUT, LIF2_SIZE, L3_SIZE_OUT, 
                                                       LIF3_SIZE};
char            layer_type[NUM_LAYERS][MAX_STR_LEN] = {"Linear", "LIF", "Linear", "LIF", "Linear", "LIF"};
cfloat_t        Beta[NUM_LAYERS]                    = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
cfloat_t        threshold[NUM_LAYERS]               = {0, 2.5, 0, 8.0, 0, 4.0};
wfloat_t        W[INPUT_SIZE * L1_SIZE_OUT  + 
                  LIF1_SIZE * L2_SIZE_OUT   +
                  LIF2_SIZE * L3_SIZE_OUT];
wfloat_t        B[L1_SIZE_OUT  + 
                  L2_SIZE_OUT  + 
                  L3_SIZE_OUT];
cfloat_t        scrachpad_memory[INPUT_SIZE + L1_SIZE_OUT];
spike_t         spike_memory[((LIF1_SIZE + LIF2_SIZE + LIF3_SIZE) / 8) + (NUM_LAYERS / 2)];
cfloat_t        mem_potential[LIF1_SIZE + LIF2_SIZE + LIF3_SIZE];
event_t*        event_list[NUM_LAYERS / 2] = {NULL, NULL, NULL};