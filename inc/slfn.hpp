/**
 * @file slfn.hpp
 * @author Lucas Silva
 */

#ifndef __SLFN_HPP__
#define __SLFN_HPP__

typedef struct _Slfn{
    /**
    * @brief Number of input nodes. Usually represents the dimension of 
    * the data on which the Neural network will produce outputs.
    */
    uint8_t input_nodes_count;
    /**
    * @brief Neuron count of the hidden layer
    */
    uint16_t hidden_neurons_count;
    /**
    * @brief Hidden Layer count. Must be set to 1 for the current implementation.
    */
    uint8_t hidden_layers_count;
    /**
    * @brief Neuron count of the output layer.
    */
    uint8_t output_neurons_count;
    /**
    * @brief Size of the training set.
    */
    uint32_t training_set_count;

    uint32_t test_set_count;
} Slfn;

#endif