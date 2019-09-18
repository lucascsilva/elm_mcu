/**
 * Copyright 2019
 * @file elm.hpp
 * @author Lucas Silva
 */

#ifndef INC_ELM_HPP_
#define INC_ELM_HPP_

#include <cstdint>
#include <cstddef>
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "slfn.hpp"

/**
 * @addtogroup elm ELM API
 * @{
 */

const float random_weights_values[] = {  // 50
  0.629447372786358,
  0.811583874151238,
  -0.746026367412988,
  0.826751712278039,
  0.264718492450819,
  -0.804919190001181,
  -0.443003562265903,
  0.0937630384099677,
  0.915013670868595,
  0.929777070398553,
  -0.684773836644903,
  0.941185563521231,
  0.914333896485891,
  -0.0292487025543176,
  0.600560937777600,
  -0.716227322745569,
  -0.156477434747450,
  0.831471050378134,
  0.584414659119109,
  0.918984852785806,
  0.311481398313174,
  -0.928576642851621,
  0.698258611737554,
  0.867986495515101,
  0.357470309715547,
  0.515480261156667,
  0.486264936249832,
  -0.215545960931664,
  0.310955780355113,
  -0.657626624376877,
  -0.42659852458743,
  0.129450486099479,
  0.296179066498458,
  0.191191332858144,
  -0.154692137784834,
  0.446816667161595,
  0.0201903182114772,
  0.453813025197927,
  -0.426404364191800,
  -0.292968051926926,
  0.275027813981967,
  0.414187821243774,
  0.282550647710300,
  -0.204465803835124,
  -0.348154277685515,
  0.347910522313782,
  0.284854591093150,
  -0.229168497848683,
  -0.272189295183866,
  -0.178976782911049,
};


const float random_bias_values[] = {  // 50
  -0.630367359751728,
  0.809761937359786,
  0.959496756712170,
  -0.122260053747794,
  -0.777761553118803,
  -0.483870608175866,
  -0.182560307774896,
  0.189792148017229,
  -0.475576504438309,
  0.205686178764166,
  0.422431560867366,
  -0.556506531965520,
  -0.765164698288388,
  -0.406648253563346,
  -0.362443396148235,
  -0.151666480572386,
  0.0157165693222363,
  -0.828968405819912,
  -0.475035530603335,
  0.602029245539478,
  -0.941559444875707,
  0.857708278956089,
  0.460661725710906,
  -0.0227820523928417,
  0.157050122046878,
  -0.525432840456957,
  -0.0823023436401378,
  0.926177078573826,
  0.0936114374779360,
  0.0422716616080030,
  -0.354267900503709,
  0.0850436152687155,
  -0.426638309919533,
  0.322326222185306,
  0.222902973268465,
  0.425858038017486,
  -0.00736140153821885,
  0.154882898449669,
  0.390123477961687,
  0.0385255744690212,
  -0.217794835441778,
  0.475957517879964,
  -0.463574484474520,
  -0.173755426939688,
  0.473013623892820,
  -0.134967374694253,
  -0.190850381104083,
  -0.379087615419372,
  0.415765704040790,
  -0.364521794318498
};

class Elm  {
 private:
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
  * @brief Fills the bias of each neuron in hidden layer with pre-stored random values.
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
  void HiddenLayerOutput(const gsl_matrix_float* samples, gsl_matrix_float* hidden_layer_outputs, const Slfn* network);
  /**
   * @brief 
   * 
   * @param 
   * @param  
   * @return
   **/
  gsl_matrix* MoorePenrosePinv(gsl_matrix *A, const double rcond);
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
  void TrainElm(const gsl_matrix_float* batch_input, gsl_matrix_float* target, const Slfn* network);
  /**
   * @brief Calculates the output of the network for a given sample.
   * 
   * @param input Pointer to the GSL Matrix containing the sample.
   * @param output Pointer to a GSL Matrix which will store the output values.
   * @param network
   **/
  void NetworkOutput(const gsl_matrix_float* input, gsl_matrix_float* output, const Slfn* network);

  
  /**
   * @}
   */
};

#endif  // INC_ELM_HPP_



