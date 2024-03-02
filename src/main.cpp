#include <iostream>
#include "snn.hpp"
#include <../Eigen/Dense>
#include <cstdlib>
#include <vector>

using namespace std;
 
void genRandVector(vector<char>& v, unsigned int size){
    int randData;
    char value;
    for(unsigned int i = 0; i < size; i++){
        randData = rand() % 10;
        value = (randData > 4) ? 1 : 0;
        v.push_back(value);
    }
}

void printVector(vector<char>& v){
    for(auto el : v){
        cout << (int)el << endl;
    }
    cout << endl;
}

int main()
{
    srand((unsigned) time(NULL));
                                
    unsigned int dimensions[] = {   8,      /* input */
                                    16,     /* L1 */
                                    32,     /* L2 */
                                    32,     /* L3 */
                                    16,     /* L4 */
                                    4       /* L5 and output */
                                };
    float beta = 0.8;
    float threshold = 0.5;
    bool reset_type = true;
    float U0[16] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

    Leaky<float> L1(dimensions[0], dimensions[1], beta, threshold, reset_type, U0);
    L1.loadWeights();
    Leaky<float> L2(dimensions[1], dimensions[2], beta, threshold, reset_type, U0);
    L2.loadWeights();
    Leaky<float> L3(dimensions[2], dimensions[3], beta, threshold, reset_type, U0);
    L3.loadWeights();
    Leaky<float> L4(dimensions[3], dimensions[4], beta, threshold, reset_type, U0);
    L4.loadWeights();
    Leaky<float> L5(dimensions[4], dimensions[5], beta, threshold, reset_type, U0);
    L5.loadWeights();

    // Input value
    vector<char> spikes;
    genRandVector(spikes, dimensions[0]);
    printVector(spikes);

    L1.writeInput(spikes);
    L1.computeOutput();
    L1.readOutput(spikes);

    L2.writeInput(spikes);
    L2.computeOutput();
    L2.readOutput(spikes);

    L3.writeInput(spikes);
    L3.computeOutput();
    L3.readOutput(spikes);

    L4.writeInput(spikes);
    L4.computeOutput();
    L4.readOutput(spikes);

    L5.writeInput(spikes);
    L5.computeOutput();
    L5.readOutput(spikes);

    printVector(spikes);
    return 0;
}