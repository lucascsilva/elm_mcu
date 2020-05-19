/**
 * Copyright 2019
 * @file elm.hpp
 * @author Lucas Silva
 */

#ifndef INC_ELM_ELM_HPP_
#define INC_ELM_ELM_HPP_

#include <cstdint>
#include <cstddef>
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "slfn.hpp"

namespace elm {
/** Flag used in positive random weight calculation */
const uint32_t POSITIVE_WEIGHT_MASK = 1;
/** Flag used in negative random weight calculation*/
const uint32_t NEGATIVE_WEIGHT_MASK = 0;
/** Current length of the array of integers */
const uint8_t RANDOM_INTEGERS_LENGTH = 45;
/** 
 * Array of random integers from which the random weights are derived.
 */
const uint32_t RANDOM_INTEGERS[] = {
  3499211589,
  3890346747,
  545404224,
  3922919432,
  2715962282,
  418932850,
  1196140743,
  2348838240,
  4112460544,
  4144164703,
  676943032,
  4168664256,
  4111000740,
  2084672538,
  3437178442,
  609397185,
  1811450916,
  3933054133,
  3402504573,
  4120988593,
  2816384858,
  153380492,
  3646982599,
  4011470454,
  2915145293,
  3254469080,
  3191729648,
  1684602222,
  2815256102,
  735241226,
  3032444858,
  136721035,
  1189375164,
  198304613,
  417177824,
  3536724443,
  2984266213,
  1361931897,
  4081172624,
  147944790,
  1884392677,
  1638781095,
  3287869570,
  3415357570,
  802611726
};
/**
 * @class Elm elm.hpp "inc/elm/elm.hpp"
 * 
 * @brief Implementation of Extreme Learning Machine. 
 */
class Elm  {
 private:
  /**
   * @brief Stores the network configuration
   */
  Slfn network_config_;
  /**
   * @brief Weight connections from input layer to hidden layer
   */ 
  gsl_matrix_float* random_weights;
  /**
   * @brief Bias to the hidden layer neurons
   * */
  gsl_matrix_float* random_bias;
  /**
   * @brief Weight of connections from hidden layer to output layer
   */ 
  gsl_matrix_float* output_weights;
    /**
   * @brief Calculates the activation function of a neuron, in the form y = 1/(1+exp(-x)).
   * 
   * @param arg The dot product between weights and inputs to the neuron (bias included).
   * @return The value of activation function.
   */
  float ActivationFunction(float arg);
    /**
   * @brief Fills the connections to input layer to hidden layer with pre-stored random values.
   **/
  void SetRandomWeights(void);
  /**
   * @brief Calculates the output of neurons at the hidden layer.
   * 
   * This method is used during training step.
   * 
   * @param samples Pointer to the GSL Matrix containing the training set.
   * @param target Pointer to a GSL Matrix which will store the values of the hidden layer output 
   * respective to the provided samples.
   **/
  void HiddenLayerOutput(const gsl_matrix_float* samples, gsl_matrix_float* hidden_layer_outputs);
  /**
   * @brief Inverts a matrix in place
   * 
   * This method uses LU decomposition to invert a matrix. Since this feature is only
   * available in double format in GSL, the matrix is copied into a double format
   * gsl_matrix, processed and the result is transformed back to float format.
   * 
   * @param [in|out] m The matrix to be inverted.
   */ 
  void invertMatrix(gsl_matrix_float* m);


 public:
  /**
   * @brief Constructor
   * 
   * Allocates memory blocks for the weights and bias. Also, sets the values for
   * random weights and random bias
   * 
   * @param network Slfn structure containing the network configuration.
   */
  explicit Elm(const Slfn &network);
  /**
  * @brief Default destructor
  * 
  * Deallocates memory blocks used by private the members
  */
  ~Elm();
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
  void TrainElm(gsl_matrix_float* batch_input, gsl_matrix_float* target);
  /**
   * @brief Calculates the output of the network for a given sample.
   * 
   * @param input Pointer to the GSL Matrix containing the sample.
   * @param output Pointer to a GSL Matrix which will store the output values.
   **/
  void NetworkOutput(const gsl_matrix_float* input, gsl_matrix_float* output);
};
}  // namespace elm
#endif  // INC_ELM_ELM_HPP_
