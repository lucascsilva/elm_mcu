/*
* Copyright 2019 Lucas Silva
*/

#include "../inc/elm/elm.hpp"

using std::size_t;

Elm::Elm(const Slfn* network) {
  random_weights = gsl_matrix_float_alloc(network->input_nodes_count, network->hidden_neurons_count);
  random_bias = gsl_matrix_float_alloc(network->hidden_layers_count, network->hidden_neurons_count);
  // output_weights = gsl_matrix_float_alloc(network->hidden_neurons_count, network->output_neurons_count);
  SetRandomWeights();
  SetRandomBias();
}

Elm::~Elm() {
  gsl_matrix_float_free(random_weights);
  gsl_matrix_float_free(random_bias);
  gsl_matrix_float_free(output_weights);
}

void Elm::SetRandomWeights(void) {
  size_t random_integer_counter = 0;
  uint8_t i = 0;
  for (size_t col_counter = 0; col_counter < random_weights->size2; col_counter++) {
    for (size_t row_counter = 0; row_counter < random_weights->size1; row_counter++) {
      if (RANDOM_INTEGERS[random_integer_counter] & (0x00000001 << i)) {
        gsl_matrix_float_set(random_weights, row_counter, col_counter, 1);
      } else {
        gsl_matrix_float_set(random_weights, row_counter, col_counter, -1);
      }
      i++;
      if (i > 31) {
        i = 0;
        random_integer_counter++;
      }
    }
  }
}

void Elm::SetRandomBias(void) {
  size_t random_integer_counter = 0;
  uint8_t i = 0;
  for (size_t col_counter = 0; col_counter < random_bias->size2; col_counter++) {
    if (RANDOM_INTEGERS[random_integer_counter] & (0x00000001 << i)) {
        gsl_matrix_float_set(random_bias, 0, col_counter, 1);
    } else {
      gsl_matrix_float_set(random_bias, 0, col_counter, -1);
    }
    i++;
    if (i > 31) {
      i = 0;
      random_integer_counter++;
    }
  }
}

void Elm::NetworkOutput(const gsl_matrix_float* input, gsl_matrix_float* output, const Slfn* network) {
  float arg;
  float sum_arg = 0;

  gsl_matrix_float* hidden_layer_output;
  hidden_layer_output = gsl_matrix_float_alloc(network->hidden_layers_count, network->hidden_neurons_count);

  for (size_t col_counter = 0; col_counter < hidden_layer_output->size2; col_counter++) {
    for (size_t i_n_counter = 0; i_n_counter < network->input_nodes_count; i_n_counter++) {
      arg = (gsl_matrix_float_get(input, i_n_counter, 0)*gsl_matrix_float_get(random_weights, i_n_counter,
                                                                                            col_counter));
      sum_arg+=arg;
    }
    sum_arg+= gsl_matrix_float_get(random_bias, 0, col_counter);  // bias
    gsl_matrix_float_set(hidden_layer_output, 0, col_counter, ActivationFunction(sum_arg));
    gsl_matrix_float_set(hidden_layer_output, 0, col_counter, sum_arg);
    sum_arg = 0;
  }

  // output calculation
  gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, hidden_layer_output, output_weights, 0, output);

  gsl_matrix_float_free(hidden_layer_output);
}

void Elm::HiddenLayerOutput(const gsl_matrix_float* samples, gsl_matrix_float* hidden_layer_outputs,
                                                                                const Slfn* network) {
  float arg, sum_arg = 0;

  for (size_t row_counter = 0; row_counter < hidden_layer_outputs->size1; row_counter++) {
    for (size_t col_counter = 0; col_counter < hidden_layer_outputs->size2; col_counter++) {
      for (size_t i_n_counter = 0; i_n_counter < network->input_nodes_count; i_n_counter++) {
        arg = (gsl_matrix_float_get(samples, i_n_counter, row_counter)*gsl_matrix_float_get(random_weights, i_n_counter,
                                                                                                          col_counter));
        sum_arg+=arg;
      }
      sum_arg+=gsl_matrix_float_get(random_bias, 0, col_counter);  // bias
      gsl_matrix_float_set(hidden_layer_outputs, row_counter, col_counter, ActivationFunction(sum_arg));
      gsl_matrix_float_set(hidden_layer_outputs, row_counter, col_counter, sum_arg);
      sum_arg = 0;
    }
  }
}

float Elm::ActivationFunction(float arg) {
  return 1/(1+exp(-arg));
}

void Elm::TrainElm(gsl_matrix_float* batch_input, gsl_matrix_float* target, const Slfn* network) {
  gsl_matrix_float *hidden_layer_outputs;

  // gsl_matrix_float *h_pseudo_inverse;
  double C = 1024;

  hidden_layer_outputs = gsl_matrix_float_alloc(network->training_set_count, network->hidden_neurons_count);
  // h_pseudo_inverse = gsl_matrix_float_alloc(network->hidden_neurons_count, network->training_set_count);

  HiddenLayerOutput(batch_input, hidden_layer_outputs, network);
  gsl_matrix_float_free(batch_input);
  // gsl_matrix_float_free(target);


  if (network->training_set_count > 10*network->hidden_neurons_count) {
    gsl_matrix_float* reg = gsl_matrix_float_alloc(network->hidden_neurons_count, network->hidden_neurons_count);
    gsl_matrix_float_set_identity(reg);
    gsl_matrix_float_scale(reg, (1.0/C));
    gsl_blas_sgemm(CblasTrans, CblasNoTrans, 1, hidden_layer_outputs, hidden_layer_outputs, 1, reg);
    invertMatrix(reg);
    output_weights = gsl_matrix_float_alloc(network->hidden_neurons_count, network->output_neurons_count);
    gsl_matrix_float *aux = gsl_matrix_float_alloc(network->hidden_neurons_count, network->output_neurons_count);
    gsl_blas_sgemm(CblasTrans, CblasNoTrans, 1, hidden_layer_outputs, target, 0, aux);
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, reg, aux, 0, output_weights);
    gsl_matrix_float_free(aux);
    gsl_matrix_float_free(reg);
  } else {
    gsl_matrix_float* reg = gsl_matrix_float_alloc(network->training_set_count, network->training_set_count);
    gsl_matrix_float_set_identity(reg);
    gsl_matrix_float_scale(reg, (1.0/C));
    gsl_blas_sgemm(CblasNoTrans, CblasTrans, 1, hidden_layer_outputs, hidden_layer_outputs, 1, reg);
    invertMatrix(reg);
    gsl_matrix_float *aux = gsl_matrix_float_alloc(network->training_set_count, network->output_neurons_count);
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, reg, target, 0, aux);
    output_weights = gsl_matrix_float_alloc(network->hidden_neurons_count, network->output_neurons_count);
    gsl_blas_sgemm(CblasTrans, CblasNoTrans, 1, hidden_layer_outputs, aux, 0, output_weights);
    gsl_matrix_float_free(aux);
    gsl_matrix_float_free(reg);
  }
  // output weights calculation
  gsl_matrix_float_free(hidden_layer_outputs);
}

void Elm::invertMatrix(gsl_matrix_float* m) {
  gsl_matrix *m_double = gsl_matrix_alloc(m->size1, m->size2);
  gsl_matrix *m_double_inv = gsl_matrix_alloc(m->size1, m->size2);
  int signum;

  for (uint16_t row = 0; row < m_double->size1; row++) {
    for (uint16_t col = 0; col < m_double->size2; col++) {
      gsl_matrix_set(m_double, row, col, static_cast<double>(gsl_matrix_float_get(m, row, col)));
    }
  }
  gsl_permutation *p = gsl_permutation_alloc(m->size1);
  gsl_linalg_LU_decomp(m_double, p, &signum);
  gsl_linalg_LU_invert(m_double, p, m_double_inv);
  gsl_permutation_free(p);
  gsl_matrix_free(m_double);
  for (uint16_t row = 0; row < m_double_inv->size1; row++) {
    for (uint16_t col = 0; col < m_double_inv->size2; col++) {
      gsl_matrix_float_set(m, row, col, static_cast<float>(gsl_matrix_get(m_double_inv, row, col)));
    }
  }
}
