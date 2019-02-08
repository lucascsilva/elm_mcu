#include "organizer.hpp"
#include "elm.hpp" 

Organizer::Organizer()
{
    samples = gsl_matrix_alloc(NUM_INPUT_NEURONS,NUM_SAMPLES);
    target = gsl_matrix_calloc(NUM_SAMPLES,NUM_OUTPUT_NEURONS); 
}

Organizer::~Organizer()
{
    gsl_matrix_free(samples);
    gsl_matrix_free(target);
}

void Organizer::storeSamples()
{

}

void Organizer::setTarget()
{

}

gsl_matrix* Organizer::getSample(void)
{
    return 0;
}

gsl_matrix* Organizer::getTarget(void)
{
    return 0;
}

