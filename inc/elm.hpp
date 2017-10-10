#ifndef __ELM_HPP__
#define __ELM_HPP__

#include "matrix_manipulation.h"

#define NUM_INPUT_NEURONS   3
#define NUM_HIDDEN_NEURONS  10
#define NUM_HIDDEN_LAYERS   1
#define NUM_OUTPUT_NEURONS  4
#define NUM_SAMPLES         20


class Elm 
{
    private:

    gsl_matrix_float* random_weights;
    gsl_matrix_float* random_bias;
    gsl_matrix_float* output_weights;
    gsl_matrix_float* hidden_layer_outputs;
    
    public:
    
    Elm();
    float ActivationFunction(float arg);
    void TrainElm(const gsl_matrix_float &batch_input, const gsl_matrix_float &target);
    void NetworkOutput(const gsl_matrix_float &input, gsl_matrix_float* output);
    void SetRandomWeights(const float *weights);
    void HiddenLayerOutput(const gsl_matrix_float* samples);
    
};

#endif