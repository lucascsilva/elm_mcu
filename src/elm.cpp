/*
* Copyright 2019 Lucas Silva
*/

#include "elm.hpp"
#include "slfn.hpp"

Elm::Elm(const Slfn* network) {
  random_weights = gsl_matrix_alloc(network->input_nodes_count, network->hidden_neurons_count);
  random_bias = gsl_matrix_alloc(network->hidden_layers_count, network->hidden_neurons_count);
  output_weights = gsl_matrix_alloc(network->hidden_neurons_count, network->output_neurons_count);
  SetRandomWeights();
  SetRandomBias();
}

Elm::~Elm() {
  gsl_matrix_free(random_weights);
  gsl_matrix_free(random_bias);
  gsl_matrix_free(output_weights);
}

void Elm::SetRandomWeights(void) {
  size_t weight_counter = 0;

  for (size_t col_counter = 0; col_counter < random_weights->size2; col_counter++) {
    for (size_t row_counter = 0; row_counter < random_weights->size1; row_counter++) {
      gsl_matrix_set(random_weights, row_counter, col_counter, random_weights_values[weight_counter++]);
    }
  }
}

void Elm::SetRandomBias(void) {
  size_t bias_counter = 0;

  for (size_t col_counter = 0; col_counter < random_bias->size2; col_counter++) {
    gsl_matrix_set(random_bias, 0, col_counter, random_bias_values[bias_counter++]);
  }
}

void Elm::NetworkOutput(const gsl_matrix* input, gsl_matrix* output, const Slfn* network)
{
  double arg;
  double sum_arg = 0;

  gsl_matrix* hidden_layer_output;
  hidden_layer_output = gsl_matrix_alloc(network->hidden_layers_count, network->hidden_neurons_count);

  for (size_t col_counter = 0; col_counter < hidden_layer_output->size2; col_counter++) {
    for (size_t i_n_counter = 0; i_n_counter < network->input_nodes_count; i_n_counter++) {
      arg = (gsl_matrix_get(input, i_n_counter, 0)*gsl_matrix_get(random_weights, i_n_counter, col_counter));
      sum_arg+=arg;
    }
    sum_arg+= gsl_matrix_get(random_bias, 0, col_counter);  // bias
    gsl_matrix_set(hidden_layer_output, 0, col_counter, ActivationFunction(sum_arg));
    gsl_matrix_set(hidden_layer_output, 0, col_counter, sum_arg);
    sum_arg = 0;
  }

  for (size_t col_counter = 0; col_counter < hidden_layer_output->size2; col_counter++) {
      arg = gsl_matrix_get(hidden_layer_output, 0, col_counter);
      gsl_matrix_set(hidden_layer_output, 0, col_counter, ActivationFunction(arg));
  }

  // output calculation
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, hidden_layer_output, output_weights, 0, output);

  gsl_matrix_free(hidden_layer_output);
}

void Elm::HiddenLayerOutput(const gsl_matrix* samples, gsl_matrix* hidden_layer_outputs, const Slfn* network) {
  float arg, sum_arg = 0;

  for (size_t row_counter = 0; row_counter < hidden_layer_outputs->size1; row_counter++) {
    for (size_t col_counter = 0; col_counter < hidden_layer_outputs->size2; col_counter++) {
      for (size_t i_n_counter = 0; i_n_counter < network->input_nodes_count; i_n_counter++) {
        arg = (gsl_matrix_get(samples, i_n_counter, row_counter)*gsl_matrix_get(random_weights, i_n_counter,
                                                                                col_counter));
        sum_arg+=arg;
      }
      sum_arg+=gsl_matrix_get(random_bias, 0, col_counter);  // bias
      gsl_matrix_set(hidden_layer_outputs, row_counter, col_counter, ActivationFunction(sum_arg));
      gsl_matrix_set(hidden_layer_outputs, row_counter, col_counter, sum_arg);
      sum_arg = 0;
    }
  }
}

double Elm::ActivationFunction(double arg)
{
    return 1/(1+exp(-arg));
}

void Elm::TrainElm(const gsl_matrix* batch_input, const gsl_matrix* target, const Slfn* network) {
  gsl_matrix *hidden_layer_outputs;
  gsl_matrix *h_pseudo_inverse;

  hidden_layer_outputs = gsl_matrix_alloc(network->training_set_count, network->hidden_neurons_count);
  h_pseudo_inverse = gsl_matrix_alloc(network->hidden_neurons_count, network->training_set_count);

  HiddenLayerOutput(batch_input, hidden_layer_outputs, network);

  h_pseudo_inverse = MoorePenrosePinv(hidden_layer_outputs, 1e-6);
  // output weights calculation
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, h_pseudo_inverse, target, 0, output_weights);

  gsl_matrix_free(hidden_layer_outputs);
  gsl_matrix_free(h_pseudo_inverse);
}


gsl_matrix* Elm::MoorePenrosePinv(gsl_matrix *A, const double rcond) {

	gsl_matrix *V, *Sigma_pinv, *U, *A_pinv;
	gsl_matrix *_tmp_mat = NULL;
	gsl_vector *_tmp_vec;
	gsl_vector *u;
	double x, cutoff;
	size_t i, j;
	unsigned int n = A->size1;
	unsigned int m = A->size2;
	bool was_swapped = false;


	if (m > n) {
		/* libgsl SVD can only handle the case m <= n - transpose matrix */
		was_swapped = true;
		_tmp_mat = gsl_matrix_alloc(m, n);
		gsl_matrix_transpose_memcpy(_tmp_mat, A);
		A = _tmp_mat;
		i = m;
		m = n;
		n = i;
	}

	/* do SVD */
	V = gsl_matrix_alloc(m, m);
	u = gsl_vector_alloc(m);
	_tmp_vec = gsl_vector_alloc(m);
	gsl_linalg_SV_decomp(A, V, u, _tmp_vec);
	gsl_vector_free(_tmp_vec);

	/* compute Σ⁻¹ */
	Sigma_pinv = gsl_matrix_alloc(m, n);
	gsl_matrix_set_zero(Sigma_pinv);
	cutoff = rcond * gsl_vector_max(u);

	for (i = 0; i < m; ++i) {
		if (gsl_vector_get(u, i) > cutoff) {
			x = 1. / gsl_vector_get(u, i);
		}
		else {
			x = 0.;
		}
		gsl_matrix_set(Sigma_pinv, i, i, x);
	}

	/* libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros */
	U = gsl_matrix_alloc(n, n);
	gsl_matrix_set_zero(U);

	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j) {
			gsl_matrix_set(U, i, j, gsl_matrix_get(A, i, j));
		}
	}

	if (_tmp_mat != NULL) {
		gsl_matrix_free(_tmp_mat);
	}

	/* two dot products to obtain pseudoinverse */
	_tmp_mat = gsl_matrix_alloc(m, n);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., V, Sigma_pinv, 0., _tmp_mat);

	if (was_swapped) {
		A_pinv = gsl_matrix_alloc(n, m);
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., U, _tmp_mat, 0., A_pinv);
	}
	else {
		A_pinv = gsl_matrix_alloc(m, n);
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., _tmp_mat, U, 0., A_pinv);
	}

	gsl_matrix_free(_tmp_mat);
	gsl_matrix_free(U);
	gsl_matrix_free(Sigma_pinv);
	gsl_vector_free(u);
	gsl_matrix_free(V);

	return A_pinv;
}





