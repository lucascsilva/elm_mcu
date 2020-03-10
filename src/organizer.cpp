/*
 * Copyright 2019
 * Lucas Silva
 */
#include "../inc/elm/organizer.hpp"

Organizer::Organizer(const Slfn &network_config)
: network_config_(network_config),
sample_count(0),
target_count(0),
samples_count(0),
targets_count(0),
result_count(0) {
  sample = new float[network_config_.input_nodes_count];
  target = new float[network_config_.output_neurons_count];
  result = new float[network_config_.test_set_count*network_config_.output_neurons_count];
  training_set = gsl_matrix_float_alloc(network_config_.input_nodes_count, network_config_.training_set_count);
  targets = gsl_matrix_float_alloc(network_config_.training_set_count, network_config_.output_neurons_count);
  test_sample = gsl_matrix_float_alloc(network_config_.input_nodes_count, 1);
}

Organizer::~Organizer() {
  gsl_matrix_float_free(training_set);
  gsl_matrix_float_free(targets);
  gsl_matrix_float_free(test_sample);
  delete sample;
  delete target;
  delete result;
}

void Organizer::buildSample(float value, Mode mode) {
  sample[sample_count++] = value;
  if (sample_count == network_config_.input_nodes_count) {
    storeSample(mode);
    sample_count = 0;
  }
}

void Organizer::buildTarget(float value) {
  target[target_count++] = value;
  if (target_count == network_config_.output_neurons_count) {
    setTarget();
    target_count = 0;
  }
}

void Organizer::storeSample(Mode mode) {
  for (uint16_t row = 0; row < network_config_.input_nodes_count; row++) {
    switch (mode) {
      case TRAIN:
          gsl_matrix_float_set(training_set, row, samples_count, sample[row]);
          break;
      case TEST:
          gsl_matrix_float_set(test_sample, row, 0, sample[row]);
          break;
    }
  }
  samples_count++;
}

void Organizer::setTarget(void) {
  for (uint8_t column = 0; column < network_config_.output_neurons_count; column++) {
    gsl_matrix_float_set(targets, targets_count, column, target[column]);
  }
  targets_count++;
}

uint16_t Organizer::getSampleCount(void) {
  return sample_count;
}

uint16_t Organizer::getTargetCount(void) {
  return targets_count;
}

uint16_t Organizer::getSamplesCount(void) {
  return samples_count;
}

uint16_t Organizer::getTargetsCount(void) {
  return targets_count;
}

gsl_matrix_float* Organizer::getSamples(void) {
  return training_set;
}

gsl_matrix_float* Organizer::getTargets(void) {
  return targets;
}

gsl_matrix_float* Organizer::getTestSample(void) {
  return test_sample;
}

void Organizer::resetSamplesCount(void) {
  samples_count = 0;
}

void Organizer::setResult(float value) {
  result[result_count++] = value;
}

float Organizer::getResult(int index) {
  return result[index];
}

void Organizer::resetResultCount(void) {
  result_count = 0;
}
