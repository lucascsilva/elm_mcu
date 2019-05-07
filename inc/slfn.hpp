/**
 * @file slfn.hpp
 * @author Lucas Silva
 */

#ifndef __SLFN_HPP__
#define __SLFN_HPP__

typedef enum _NeuronType
{
    ADDITIVE=0,
    SIGMOID
}NeuronType;

typedef struct _Slfn{
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

    uint32_t test_set_count;

    NeuronType output_neuron_type;
}Slfn;

#endif