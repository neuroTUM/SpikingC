# SpikingC
SNN implementation in C

To control different tests and implementations macros in SNNconfig.h have to be (un)defined.

To test the code with just a single input TEST macro has to be defined. In this case, intermediate values will be compared against the reference values generated with a pytorch model. It is advised to use this one when debugging, since it gives the most amount of infomation.

To calculate accuracy on the testset DATALOADER has to be defined. 

To load data from the binary files BINARY_IMPLEMENTATION has to be defined. If this is not the case CSV files will be used. Currently, BINARY_IMPLEMENTATION + TEST doesn't work, although BINARY_IMPLEMENTATION works when DATALOADER is defined.

To measure the execution time just run the program as follows(linux only):
time ./SpikingC