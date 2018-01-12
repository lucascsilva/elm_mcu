#include "elm.hpp"

Elm::Elm()
{
    random_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS,NUM_INPUT_NEURONS);
    random_bias = gsl_matrix_alloc(NUM_HIDDEN_LAYERS, NUM_HIDDEN_NEURONS);
    output_weights = gsl_matrix_alloc(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS);
    SetRandomWeights();
    SetRandomBias();
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
        -0.657626624376877
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

void Elm::SetRandomBias(void)
{
    double random_bias_values[]=
    {
        0.412092176039218,
        -0.936334307245159,
        -0.446154030078220,
        -0.907657218737692,
        -0.805736437528305,
        0.646915656654585,
        0.389657245951634,
        -0.365801039878279,
        0.900444097676710,
        -0.931107838994183
    };

    size_t bias_counter=0;

    for(size_t col_counter=0; col_counter<random_bias->size2; col_counter++)
    {
        gsl_matrix_set(random_bias, 0, col_counter, random_bias_values[bias_counter++]);
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
            gsl_matrix_set(hidden_layer_outputs,row_counter, col_counter, ActivationFunction(sum_arg));
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


gsl_matrix* moore_penrose_pinv(gsl_matrix *A, const double rcond) 
{

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




