#include "elm.hpp"


Elm::Elm()
{
    random_weights = gsl_matrix_float_alloc(NUM_HIDDEN_NEURONS,NUM_INPUT_NEURONS);
    random_bias = gsl_matrix_float_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    output_weights = gsl_matrix_float_alloc(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS);
    hidden_layer_outputs = gsl_matrix_float_alloc(NUM_SAMPLES, NUM_HIDDEN_NEURONS);
}

void Elm::SetRandomWeights(const float *weights)
{
    size_t weight_counter=0;

    for(size_t row_counter=0; row_counter<random_weights->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<random_weights->size2; col_counter++)
        {
            gsl_matrix_float_set(random_weights,row_counter, col_counter, weights[weight_counter++]);
        }   
    }
}

void Elm::NetworkOutput(const gsl_matrix_float &input, gsl_matrix_float* output)
{

}

void Elm::HiddenLayerOutput(const gsl_matrix_float* samples)
{
    
    float arg,sum_arg=0;

    for(size_t row_counter=0; row_counter<hidden_layer_outputs->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<hidden_layer_outputs->size2; col_counter++)
        {
            for(size_t counter=0; counter<NUM_INPUT_NEURONS;counter++)
            {
                arg=(gsl_matrix_float_get(samples,counter,row_counter)*gsl_matrix_float_get(random_weights,counter,col_counter));
                sum_arg+=arg;
            }
            sum_arg+=gsl_matrix_float_get(random_bias,1,col_counter); //bias
            gsl_matrix_float_set(hidden_layer_outputs,row_counter, col_counter, ActivationFunction(arg));
        }   
    }

}

float Elm::ActivationFunction(float arg)
{
    return 1/(1+exp(-arg));
}







