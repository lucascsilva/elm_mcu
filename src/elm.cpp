#include "elm.hpp"


Elm::Elm()
{
    random_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS,NUM_INPUT_NEURONS);
    random_bias = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    output_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS);
}

Elm::~Elm()
{
    gsl_matrix_free(random_weights);
    gsl_matrix_free(random_bias);
    gsl_matrix_free(output_weights);

}

void Elm::SetRandomWeights(const double *weights)
{
    size_t weight_counter=0;

    for(size_t row_counter=0; row_counter<random_weights->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<random_weights->size2; col_counter++)
        {
            gsl_matrix_set(random_weights, row_counter, col_counter, weights[weight_counter++]);
        }   
    }
}

void Elm::NetworkOutput(const gsl_matrix* input, gsl_matrix* output)
{
    double arg;
    
    gsl_matrix* hidden_layer_output;
    hidden_layer_output = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,input,random_weights,0,hidden_layer_output);
    gsl_matrix_add(hidden_layer_output, random_bias);

    for (size_t col_counter=0; col_counter<hidden_layer_output->size2; col_counter++)
    {
        arg=gsl_matrix_get(hidden_layer_output,1,col_counter);
        gsl_matrix_set(hidden_layer_output,1,col_counter,ActivationFunction(arg));
    }

    //output calculation
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,hidden_layer_output,output_weights,0,output);
    
    gsl_matrix_free(hidden_layer_output);

}

void Elm::HiddenLayerOutput(const gsl_matrix* samples, gsl_matrix* hidden_layer_outputs)
{
    
    float arg,sum_arg=0;

    for(size_t row_counter=0; row_counter<hidden_layer_outputs->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<hidden_layer_outputs->size2; col_counter++)
        {
            for(size_t counter=0; counter<NUM_INPUT_NEURONS;counter++)
            {
                arg=(gsl_matrix_get(samples,counter,row_counter)*gsl_matrix_get(random_weights,counter,col_counter));
                sum_arg+=arg;
            }
            sum_arg+=gsl_matrix_get(random_bias,1,col_counter); //bias
            gsl_matrix_set(hidden_layer_outputs,row_counter, col_counter, ActivationFunction(arg));
        }   
    }

}

double Elm::ActivationFunction(double arg)
{
    return 1/(1+exp(-arg));
}

void Elm::TrainElm(const gsl_matrix* batch_input, const gsl_matrix* target)
{
    gsl_matrix *hidden_layer_outputs;
    gsl_matrix *h_pseudo_inverse;

    hidden_layer_outputs = gsl_matrix_alloc(NUM_SAMPLES, NUM_HIDDEN_NEURONS);
    h_pseudo_inverse = gsl_matrix_alloc(NUM_HIDDEN_NEURONS, NUM_SAMPLES);

    Elm::HiddenLayerOutput(batch_input, hidden_layer_outputs);

    h_pseudo_inverse = moore_penrose_pinv(hidden_layer_outputs, 1e-6);
    //output weights calculation
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,h_pseudo_inverse,target,0,output_weights);

    gsl_matrix_free(hidden_layer_outputs);
    gsl_matrix_free(h_pseudo_inverse);
}







