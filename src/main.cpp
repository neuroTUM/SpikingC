#include <iostream>
#include <chrono>
#include "../include/Model.hpp"

using namespace std;

int main()
{   
    auto start = chrono::high_resolution_clock::now();
    #ifdef LOAD
    loadWeights();
    parseReferenceJSON();
    #endif
    /************************************************************************************************/
    Model SNN;
    cfloat_array_t In;

    /* Reset membrane potentials and spikes */
    SNN.resetState();
    
    for(unsigned int i = 0; i < 31; i++){
        
        /* Load intput for this time step */
        for(unsigned int j = 0; j < INPUT_SIZE; j++){
            scrachpad_memory[j] = output_ref["inputs"][i][0][j];
        }
        In.ptr  = scrachpad_memory;
        In.size = INPUT_SIZE;

        /* Perform computations */
        SNN.run(&In);
    }

    cout << "Predicted class is " << SNN.predict() << endl;

    /************************************************************************************************/
    auto end = chrono::high_resolution_clock::now();
    double time_taken = 
    chrono::duration_cast<chrono::nanoseconds>(end - start).count();
 
    time_taken *= 1e-9;
 
    cout << "Time taken by program is : " << fixed 
         << time_taken << setprecision(9);
    cout << " sec" << endl;    

    return 0;
}