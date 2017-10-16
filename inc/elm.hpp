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

    gsl_matrix* random_weights;
    gsl_matrix* random_bias;
    gsl_matrix* output_weights;
    
    public:
    
    Elm();
    ~Elm();
    double ActivationFunction(double arg);
    void TrainElm(const gsl_matrix* batch_input, const gsl_matrix* target);
    void NetworkOutput(const gsl_matrix* input, gsl_matrix* output);
    void SetRandomWeights(const double *weights);
    void HiddenLayerOutput(const gsl_matrix* samples, gsl_matrix* hidden_layer_outputs);
    
};

#endif