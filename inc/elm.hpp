#ifndef __ELM_HPP__
#define __ELM_HPP__

#include "matrix_manipulation.h"

#define NUM_INPUT_NEURONS
#define NUM_HIDDEN_NEURONS
#define NUM_OUTPUT_NEURONS


class Elm 
{
    private:

    gsl_matrix random_weights;
    gsl_matrix output_weights;

    public:

    float ActivationFunction(float arg);
    void TrainElm(const gsl_matrix &batch_input, const gsl_matrix &target);
    void NetworkOutput(const gsl_matrix &input, gsl_matrix* output);

};

#endif