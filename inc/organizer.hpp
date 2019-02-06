/**
 * @file elm.hpp
 * @author Lucas Silva
 */

#ifndef __ORGANIZER_HPP__
#define __ORGANIZER_HPP__

#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"

class Organizer
{
    private:
    gsl_matrix* samples;
    gsl_matrix* target;
    public:
    void storeSamples();
    void setTarget();
    gsl_matrix* getSample(void);
    gsl_matrix* getTarget(void);
}

#endif