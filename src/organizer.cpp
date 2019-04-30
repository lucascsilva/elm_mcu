#include "organizer.hpp"
#include "slfn.hpp"

Organizer::Organizer(const Slfn* network)
:sample_count(0),
target_count(0),
samples_count(0),
targets_count(0),
result_count(0)
{
    sample = new float[network->input_nodes_count];
    target = new float[network->output_neurons_count];
    result = new float[network->test_set_count*network->output_neurons_count];
    training_set = gsl_matrix_alloc(network->input_nodes_count, network->training_set_count);
    targets = gsl_matrix_alloc(network->training_set_count, network->output_neurons_count);
    test_sample = gsl_matrix_alloc(network->input_nodes_count, 1);
}

Organizer::~Organizer()
{
    gsl_matrix_free(training_set);
    gsl_matrix_free(targets);
    gsl_matrix_free(test_sample);
    delete sample;
    delete target;
    delete result;
}


void Organizer::buildSample(float value, Mode mode, const Slfn* network)
{
    sample[sample_count++]=value;
    if(sample_count == network->input_nodes_count)
    {
        storeSample(mode, network);
        sample_count=0;
    }      
}

void Organizer::buildTarget(float value, const Slfn* network)
{
    target[target_count++]=value;
    if(target_count == network->output_neurons_count)
    {
        setTarget(network);
        target_count=0;
    }       
}

void Organizer::storeSample(Mode mode, const Slfn* network)
{
    for (uint16_t row = 0; row < network->input_nodes_count; row++)
    {
        switch(mode)
        {
            case TRAIN:
                gsl_matrix_set(training_set, row, samples_count, sample[row]);
                break;
            case TEST:
                gsl_matrix_set(test_sample, row, 0, sample[row]);
                break;
        }
    }  
    samples_count++; 
}

void Organizer::setTarget(const Slfn* network)
{
    for (uint8_t column = 0; column < network->output_neurons_count; column++)
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
    return training_set;
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

void Organizer::setResult(float value)
{
    result[result_count++]=value;
}

float Organizer::getResult(int index)
{
   return result[index];
}

void Organizer::resetResultCount(void)
{
    result_count=0;
}