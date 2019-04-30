/**
 * @file elm.hpp
 * @author Lucas Silva
 */

#ifndef __ELM_HPP__
#define __ELM_HPP__

#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "slfn.hpp"

/**
 * @addtogroup elm ELM API
 * @{
 */

class Elm 
{
    private:
    /**
     * @brief Weight connections from input layer to hidden layer
     */ 
    gsl_matrix* random_weights;
    /**
     * @brief Bias to the hidden layer neurons
     * */
    gsl_matrix* random_bias;
    /**
     * @brief Weight of connections from hidden layer to output layer
     */ 
    gsl_matrix* output_weights;
    
    public:
    /**
     * @brief Default Constructor
     * 
     * Allocates memory blocks for the weights and bias. Also, sets the values for
     * random weights and random bias
     * 
     * @param network 
     */
    Elm(const Slfn* network);
    /**
    * @brief Default destructor
    * 
    * Deallocates memory blocks used by private the members
    */
    ~Elm();
    /**
     * @brief Calculates the activation function of a neuron, in the form y = 1/(1+exp(-x)).
     * 
     * @param arg The dot product between weights and inputs to the neuron (bias included).
     * @return The value of activation function.
     */
    double ActivationFunction(double arg);
    /**
     * @brief Calculates the output weight values.
     * 
     * Calculation is made by multiplication of pseudoinverse of hidden layer output matrix H by target:
     * 
     * output_wieghts = pinv(H)*target
     * 
     * @param batch_input Pointer to the GSL Matrix containing the samples set.
     * @param target Pointer to a GSL Matrix containing the wanted outputs respective to the training set.
     * 
     */
    void TrainElm(const gsl_matrix* batch_input, const gsl_matrix* target, const Slfn* network);
    /**
     * @brief Calculates the output of the network for a given sample.
     * 
     * @param input Pointer to the GSL Matrix containing the sample.
     * @param output Pointer to a GSL Matrix which will store the output values.
     * @param network
     **/
    void NetworkOutput(const gsl_matrix* input, gsl_matrix* output, const Slfn* network);
    /**
     * @brief Fills the connections to input layer to hidden layer with pre-calculated random values.
     **/
    void SetRandomWeights(void);
    /**
    * @brief Fills the bias of each neuron in hidden layer with pre-calculated random values.
    **/
    void SetRandomBias(void);
    /**
     * @brief Calculates the output of neurons at the hidden layer.
     * 
     * This method is used during training step.
     * 
     * @param samples Pointer to the GSL Matrix containing the training set.
     * @param target Pointer to a GSL Matrix which will store the values of the hidden layer output 
     * respective to the provided samples.
     * @param network
     **/
    void HiddenLayerOutput(const gsl_matrix* samples, gsl_matrix* hidden_layer_outputs, const Slfn* network);
    /**
     * @brief 
     * 
     * @param 
     * @param  
     * @return
     **/
    gsl_matrix* MoorePenrosePinv(gsl_matrix *A, const double rcond);
    /**
     * @}
     */

};

#endif



