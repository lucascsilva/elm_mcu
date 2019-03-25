#include "organizer.hpp"
#include "elm.hpp" 

Organizer::Organizer()
:sample_count(0),
target_count(0),
samples_count(0),
targets_count(0)
{
    samples = gsl_matrix_alloc(NUM_INPUT_NEURONS,NUM_SAMPLES);
    targets = gsl_matrix_calloc(NUM_SAMPLES,NUM_OUTPUT_NEURONS);
    test_sample = gsl_matrix_calloc(NUM_INPUT_NEURONS,1);
}

Organizer::~Organizer()
{
    gsl_matrix_free(samples);
    gsl_matrix_free(targets);
    gsl_matrix_free(test_sample);
}


void Organizer::buildSample(float value, Mode mode)
{
    sample[sample_count++]=value;
    if(sample_count == sample.size())
    {
        storeSample(mode);
        sample_count=0;
    }      
}

void Organizer::buildTarget(float value)
{
    target[target_count++]=value;
    if(target_count == target.size())
    {
        setTarget();
        target_count=0;
    }       
}

void Organizer::storeSample(Mode mode)
{
    for (uint16_t row = 0; row < NUM_INPUT_NEURONS; row++)
    {
        switch(mode)
        {
            case TRAIN:
                gsl_matrix_set(samples, row, samples_count, sample[row]);
                break;
            case TEST:
                gsl_matrix_set(test_sample, row, 0, sample[row]);
                break;
        }
    }  
    samples_count++; 
}

void Organizer::setTarget(void)
{
    for (uint8_t column = 0; column < NUM_OUTPUT_NEURONS; column++)
    {
        gsl_matrix_set(targets, targets_count, column, target[column]);
    }   
    targets_count++;
}


uint16_t Organizer::getSampleCount(void)
{
    return sample_count;
}

uint16_t Organizer::getTargetCount(void)
{
    return targets_count;
}

uint16_t Organizer::getSamplesCount(void)
{
    return samples_count;
}

uint16_t Organizer::getTargetsCount(void)
{
    return targets_count;
}

gsl_matrix* Organizer::getSamples(void)
{
    return samples;
}

gsl_matrix* Organizer::getTargets(void)
{
    return targets;
}

gsl_matrix* Organizer::getTestSample(void)
{
    return test_sample;
}

void Organizer::resetSamplesCount(void)
{
    samples_count=0;
}
