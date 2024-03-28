#include "../include/Model.h"
#include "../include/Utility.h"

int main(void)
{   
    model_t SNN;
    initModel(&SNN);

    cfloat_array_t In;
    In.size = INPUT_SIZE;
    In.ptr = scrachpad_memory;

    /* Load weights and biases */

    /* Reset the state */
    SNN.resetState_fptr(&SNN);

    for(unsigned int i = 0; i < TIME_STEPS; i++){
        /* Load the data for this time step */

        /* Run the model for one time step */
        SNN.run_fptr(&SNN, &In);
    }

    printf("Predicted class is: %d\n", SNN.predict_fptr(&SNN));
    
    /* Free memory allocated for SNN */
    SNN.clearModel_fptr(&SNN);

    return 0;
}