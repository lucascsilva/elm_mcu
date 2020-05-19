/**
 * @file slfn.hpp
 * @author Lucas Silva
 */

#ifndef INC_ELM_SLFN_HPP_
#define INC_ELM_SLFN_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>

using std::size_t;

namespace elm {
/**
 * @enum NeuronType slfn.hpp "inc/elm/slfn.hpp"
 * 
 * @brief Types for a neuron.
 */
enum NeuronType {
  ADDITIVE = 0, /**< additive neuron \f$y=wx +b\f$*/
  SIGMOID /**< Limiter neuron \f$y=f(wx+b)\f$ where \f$f(.)\f$ is the sigmoid function$*/
};

/**
 * @struct Slfn slfn.hpp "inc/elm/slfn.hpp"
 * 
 * @brief Contains the configuration parameters for a Single Layer FeedForward Network (SLFN).
 */
struct Slfn {
  /**
  * @brief Number of input nodes. Usually represents the dimension of 
  * the data on which the Neural network will produce outputs.
  */
  size_t input_nodes_count;
  /**
  * @brief Neuron count of the hidden layer
  */
  size_t hidden_neurons_count;
  /**
  * @brief Hidden Layer count. Must be set to 1 for the current implementation.
  */
  size_t hidden_layers_count;
  /**
  * @brief Neuron count of the output layer.
  */
  size_t output_neurons_count;
  /**
  * @brief Size of the training set.
  */
  uint32_t training_set_count;
  /**
   * @brief Size of test set
   */
  uint32_t test_set_count;
  /**
   * @brief Type of activation function for neurons. Currently only additive types, 
   * so this parameter has no use for now
   */
  NeuronType output_neuron_type;
  /**
   * @brief Number of bits used to encode the random weights. More bits will provide more
   * possible random weights in the interval [-1 ,1]
   */
  uint8_t bits;
};
}  // namespace elm
#endif  // INC_ELM_SLFN_HPP_
