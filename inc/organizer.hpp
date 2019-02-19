/**
 * @file elm.hpp
 * @author Lucas Silva
 */

#ifndef __ORGANIZER_HPP__
#define __ORGANIZER_HPP__

#include <array>
#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "elm.hpp"


class Organizer
{
    private:
    std::array<float, NUM_INPUT_NEURONS> sample;
    std::array<float, NUM_OUTPUT_NEURONS> target;
    gsl_matrix* samples;
    gsl_matrix* targets;
    uint16_t sample_count;
    uint16_t target_count;
    uint16_t samples_count;
    uint16_t targets_count;

    
    public:
    Organizer();
    ~Organizer();
    void buildSample(float value);
    void buildTarget(float value);
    void storeSample(void);
    void setTarget(void);
    uint16_t getSamplesCount(void);
    uint16_t getTargetsCount(void);
    gsl_matrix* getSamples(void);
    gsl_matrix* getTargets(void);
};

#endif