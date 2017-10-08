#include "elm.hpp"


Elm::Elm()
{
    random_weights = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    output_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS);
    hidden_layer_outputs = gsl_matrix_alloc(NUM_SAMPLES, NUM_HIDDEN_NEURONS);
}

void Elm::SetRandomWeights(const float *weights)
{
    size_t weight_counter=0;
    for(size_t row_counter=0; row_counter < random_weights->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<random_weights->size2; col_counter++)
        {
            gsl_matrix_set(random_weights,row_counter, col_counter, weights[weight_counter++]);
        }   
    }
}


