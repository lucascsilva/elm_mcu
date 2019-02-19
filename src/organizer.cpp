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
}

Organizer::~Organizer()
{
    gsl_matrix_free(samples);
    gsl_matrix_free(targets);
}


void Organizer::buildSample(float value)
{
    sample[sample_count++]=value;
    if(sample_count == sample.size())
    {
        storeSample();
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

void Organizer::storeSample(void)
{
    for (uint16_t row = 0; row < NUM_INPUT_NEURONS; row++)
    {
        gsl_matrix_set(samples, row, samples_count, sample[row]);
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

