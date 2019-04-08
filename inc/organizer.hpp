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

typedef enum _Mode
{
    TRAIN = 0,
    TEST
}Mode;

typedef struct _OutputData
{
    gsl_matrix *output;
    size_t row_max;
    size_t col_max;
}OutputData;

class Organizer
{
    private:
    std::array<float, NUM_INPUT_NEURONS> sample;
    std::array<float, NUM_OUTPUT_NEURONS> target;
    std::array<float, NUM_TEST> result;
    gsl_matrix* training_set;
    gsl_matrix* targets;
    gsl_matrix* test_sample;
    uint16_t sample_count;
    uint16_t target_count;
    uint16_t samples_count;
    uint16_t targets_count;
    uint16_t result_count;

    
    public:
    Organizer();
    ~Organizer();
    void buildSample(float value, Mode mode);
    void buildTarget(float value);
    void storeSample(Mode mode);
    void setTarget(void);
    uint16_t getSampleCount(void);
    uint16_t getTargetCount(void);
    uint16_t getSamplesCount(void);
    uint16_t getTargetsCount(void);
    gsl_matrix* getSamples(void);
    gsl_matrix* getTargets(void);
    gsl_matrix* getTestSample(void);
    void resetSamplesCount(void);
    void resetResultCount(void);
    void setResult(float value);
    float getResult(int index);

};

#endif