#include "../include/SNNconfig.h"

unsigned int    layer_size[NUM_LAYERS + 1]          = {INPUT_SIZE, L1_SIZE_OUT, LIF1_SIZE, 
                                                       L2_SIZE_OUT, LIF2_SIZE, L3_SIZE_OUT, 
                                                       LIF3_SIZE};
char            layer_type[NUM_LAYERS][MAX_STR_LEN] = {"Linear", "LIF", "Linear", "LIF", "Linear", "LIF"};
char            Beta[NUM_LAYERS]                    = {1, 1, 1, 1, 1, 1};
fxp16_t         threshold[NUM_LAYERS]               = {0, 640, 0, 2048, 0, 1024};
fxp16_t         L[NUM_LAYERS]                       = {0, 320, 0, 1024, 0, 512};
fxp8_t          W[INPUT_SIZE * L1_SIZE_OUT  + 
                  LIF1_SIZE * L2_SIZE_OUT   +
                  LIF2_SIZE * L3_SIZE_OUT];
fxp8_t          B[L1_SIZE_OUT  + 
                  L2_SIZE_OUT  + 
                  L3_SIZE_OUT];
fxp16_t         scrachpad_memory[INPUT_SIZE];
fxp16_t         mem_potential[LIF1_SIZE + LIF2_SIZE + LIF3_SIZE];
event_t*        event_list = NULL;