#include "../include/Linear.h"

void clearLinear(linear_t* layer){
    for(unsigned int i = 0; i < layer->W.rows; i++)
        free(layer->W.ptr[i]);
    layer->W.ptr = NULL;
    layer->B.ptr = NULL;
}

void initLinear(linear_t* layer, unsigned int layer_num){
    layer->layer_num                        = layer_num;
    layer->W.ptr                        	= returnWeightPtr(layer_num);
    layer->W.rows                       	= layer_size[layer_num + 1];
    layer->W.cols                       	= layer_size[layer_num];
    layer->B.ptr                        	= returnBiasPtr(layer_num);
    layer->B.size                       	= layer_size[layer_num + 1];
    layer->clearLinear_fptr             	= &clearLinear;
    layer->matrixVectorMul_fptr             = &matrixVectorMul;
    layer->matrixVectorMulSparse_fptr       = &matrixVectorMulSparse;
}