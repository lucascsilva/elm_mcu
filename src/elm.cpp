#include "elm.hpp"



Elm::Elm()
{
    random_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS,NUM_INPUT_NEURONS);
    random_bias = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    output_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS);
    SetRandomWeights();
}

Elm::~Elm()
{
    gsl_matrix_free(random_weights);
    gsl_matrix_free(random_bias);
    gsl_matrix_free(output_weights);
}

void Elm::SetRandomWeights(void)
{
    double random_weights_values[]=
    {
        0.814723686393179,
        0.905791937075619,
        0,126986816293506,
        0.913375856139019,
        0.632359246225410,
        0.097540404999409,
        0.278498218867048,
        0.546881519204984,
        0.957506835434298,
        0.964888535199277,
        0.157613081677548,
        0.970592781760616,
        0.957166948242946,
        0.485375648722841,
        0.800280468888800,
        0.141886338627215,
        0.421761282626275,
        0.915735525189067,
        0.792207329559554,
        0.959492426392903,
        0.655740699156587,
        0.035711678574186,
        0.849129305868777,
        0.933993247757551,
        0.678735154857774,
        0.757740130578333,
        0.743132468124916,
        0.392227019534168,
        0.655477890177557,
        0.171186687811562
    };

    size_t weight_counter=0;

    for(size_t row_counter=0; row_counter<random_weights->size1; row_counter++)
    {
        for(size_t col_counter=0; col_counter<random_weights->size2; col_counter++)
        {
            gsl_matrix_set(random_weights, row_counter, col_counter, random_weights_values[weight_counter++]);
        }   
    }
}

void Elm::NetworkOutput(const gsl_matrix* input, gsl_matrix* output)
{
    double arg;
    
    gsl_matrix* hidden_layer_output;
    hidden_layer_output = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    
    gsl_blas_dgemm(CblasTrans,CblasTrans,1,input,random_weights,0,hidden_layer_output);
    gsl_matrix_add(hidden_layer_output, random_bias);

    for (size_t col_counter=0; col_counter<hidden_layer_output->size2; col_counter++)
    {
        arg=gsl_matrix_get(hidden_layer_output,0,col_counter);
        gsl_matrix_set(hidden_layer_output,0,col_counter,ActivationFunction(arg));
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
            for(size_t i_n_counter=0; i_n_counter<NUM_INPUT_NEURONS;i_n_counter++)
            {
                arg=(gsl_matrix_get(samples,i_n_counter,row_counter)*gsl_matrix_get(random_weights,col_counter,i_n_counter));
                sum_arg+=arg;
            }
            sum_arg+=gsl_matrix_get(random_bias,0,col_counter); //bias
            gsl_matrix_set(hidden_layer_outputs,row_counter, col_counter, ActivationFunction(arg));
            sum_arg=0;
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

    HiddenLayerOutput(batch_input, hidden_layer_outputs);

    h_pseudo_inverse = moore_penrose_pinv(hidden_layer_outputs, 1e-6);
    //output weights calculation
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,h_pseudo_inverse,target,0,output_weights);

    gsl_matrix_free(hidden_layer_outputs);
    gsl_matrix_free(h_pseudo_inverse);
}







