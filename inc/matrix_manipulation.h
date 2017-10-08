#ifndef _MATRIX_MANIPULATION_H_
#define _MATRIX_MANIPULATION_H_

#include <stdio.h>
#include <stdint.h>
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"


typedef double realtype;

#define max(a,b)		((a) > (b) ? (a) : (b))
#define min(a,b)		((a) < (b) ? (a) : (b))

gsl_matrix* moore_penrose_pinv(gsl_matrix *A, const realtype rcond);

#endif