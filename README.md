# SpikingC
SNN implementation in C

To control different tests and implementations macros in SNNconfig.h have to be (un)defined.

To test the code with just a single input TEST macro has to be defined. In this case, intermediate values will be compared against the reference values generated with a pytorch model. It is advised to use this one when debugging, since it gives the most amount of infomation.

To calculate accuracy on the testset DATALOADER has to be defined. 

To load data from the binary files BINARY_IMPLEMENTATION has to be defined. If this is not the case CSV files will be used. Currently, BINARY_IMPLEMENTATION + TEST doesn't work, although BINARY_IMPLEMENTATION works when DATALOADER is defined.

To measure the execution time just run the program as follows(linux only):
time ./SpikingC

```
├── CMakeLists.txt
├── Makefile
├── README.md
├── include
│   ├── LIF.h
│   ├── Linear.h
│   ├── Model.h
│   ├── SNNconfig.h # configuration of the application
│   └── Utility.h
├── models # saved models for inference
│   └── SNN_3L_simple_LIF_NMNIST
│       ├── intermediate_outputs
│       │   ├── fc1
│       │   │   ├── fc1_outputs_timestep_0.csv
│       │   │   │ ...
├── notebooks  # python notebooks for training, testing, prunning and quantzation
├── src
│   ├── LIF.c
│   ├── Linear.c
│   ├── Model.c
│   ├── SNNconfig.c
│   ├── Utility.c
│   └── main.c
```

| Macro       | Description                | Value |
|-------------|----------------------------|-------|
| TIME_STEPS  | Number of time steps       | 31    |
| CHANNELS    | Number of input channels   | 2     |
| HEIGHT      | Height of input images     | 34    |
| WIDTH       | Width of input images      | 34    |
| NUM_LAYERS  | Number of layers in the network | 6 |
| INPUT_SIZE  | Total size of the input layer | 2312 |
| TEST        | Does one pass of single input and test intermedia results | FALSE |
| DATALODER   | Use whole test dataset in inference mode | TRUE |

                                                                                                                                          
                                                                                                                           
                                                                                                                                     
                             @@@@                                                                                                    
                     @@@@ @@@@@@@@@                                                                                                  
                  @@@@@@@@@@@   @@@@                                                                                                 
                 @@@      @@@@@ @@@@                                                                                                 
              @@@@@@        @@@  @@                                                                                                  
             @@@ @@@  @@@    @                                                                                                       
            @@@@  @@@@@@@@     @@@@@@                                                                                                
          @@@@@@        @@     @@@@@@@                         @@  @@@@       @@@@@       @@     @@      @@  @@      @@@@@           
         @@@@     @@   @@@   @@@@@@@@@@@@@                 @@@@@@@@@@@@@@   @@ @@@@@  @@@@@@ @@@@@@  @@@@@@@@@@@  @@  @@@@@@@        
         @@@    @@@        @@@@     @@@ @@@@@@@             @@@@@  @@@@@@ @@@@  @@@@   @@@@@  @@@@@   @@@@@ @@@@ @@@@  @@@@@@@       
         @@@    @@       @@@@        @@@@   @@@@@@@@@@@     @@@@@  @@@@@@ @@@@@ @@@@   @@@@@  @@@@@   @@@@@   @@ @@@@@  @@@@@@       
         @@@@   @@     @@@@            @@@       @@@@@@@    @@@@@  @@@@@@ @@@@@@@      @@@@@  @@@@@   @@@@@      @@@@@@   @@@@       
      @@@ @@@@@@@@  @@@@       @@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@  @@@@@@  @@@@@@@@@@  @@@@@@@@@@@@   @@@@@      @@@@@@@  @@@        
     @@@ @@@@@@@  @@@@      @@@@@@@@@@@@@ @@@ @@@@@@@@@    @@@@@@@ @@@@@@@  @@@@@@@@@  @@@@@@@@@@@@@ @@@@@@@@      @@@@@@@@@         
     @@  @@@    @@@@ @@@@@@@@@@@@@@@@@@    @@@@@@@@@                                                                                 
    @@@ @@  @@@@@@@@@@@@             @@@@@@@@@@@@@@        @@@@@@@@@@@@@@@@@@@  @@@@@@@@@    @@@@@     @@@@@@@      @@@@@@@@         
    @@@ @@  @@@@@@@@@@                @@@@@@@@@@@@         @@@@  @@@@@@@  @@@@   @@@@@@@       @@       @@@@@@@    @@@@@@@@          
     @@ @@   @@@@@@@@@@@@@@@     @@ @@@@   @@@@ @@@        @@    @@@@@@@    @@   @@@@@@@       @@      @@@@@@@@@   @@@@@@@@@         
     @@@@@@      @@@@     @@@@@@@@@@@     @@@@@@@@@@@      @     @@@@@@@     @   @@@@@@@       @@      @@ @@@@@@   @ @@@@@@@         
         @@@@@      @@@@       @@@@@@@@@@@@@@@@@@@@@@@@@         @@@@@@@         @@@@@@@       @@      @@ @@@@@@@ @@  @@@@@@         
          @@@@@@@@@   @@@@      @@@@   @@@@@@@@@@@@@@@@@         @@@@@@@         @@@@@@@       @@     @@@  @@@@@@@@   @@@@@@         
         @@@@    @@  @  @@@@          @@@     @@@@@@@@@@         @@@@@@@         @@@@@@@       @@     @@    @@@@@@@   @@@@@@@        
         @@@@      @@@@   @@@@      @@@   @@@@@@                 @@@@@@@          @@@@@@       @@     @@     @@@@@    @@@@@@@        
          @@@@@ @@@@@@      @@@@@@@@@@@@@@@@                     @@@@@@@          @@@@@@      @@      @@     @@@@      @@@@@@        
            @@@@@@             @@@@@@@@@                       @@@@@@@@@@@          @@@@@@@@@@@     @@@@@@%   @@@    @@@@@@@@@@      
              @@@   @@@    @@  @@@@@@                                                                                                
              @@@@@@@@    @@@   @@@@                                                                                                 
                @@@       @@                                                                                                         
                 @@@@@@@@@@@@   @@@                                                                                                  
                   @@@@@@@@@@@  @@@                                                                                                  
                           @@@@@@@@                                                                                                  
                             @@@@@                                                                                                   
                                                                                                                                     
                                                                                                                                                                                                                         
