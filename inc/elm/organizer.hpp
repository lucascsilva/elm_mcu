/*
 * Copyright 2019
 * Lucas Silva
 */
/**
 *  @file organizer.hpp
 */

#ifndef INC_ELM_ORGANIZER_HPP_
#define INC_ELM_ORGANIZER_HPP_

#include "../inc/gsl/gsl_blas.h"
#include "../inc/gsl/gsl_linalg.h"
#include "../inc/elm/slfn.hpp"

enum Mode {
  TRAIN = 0,
  TEST
};

struct OutputData {
  gsl_matrix_float *output;
  size_t row_max;
  size_t col_max;
};

class Organizer {
 private:
  Slfn network_config_;
  float* sample;
  float* target;
  float* result;
  gsl_matrix_float* training_set;
  gsl_matrix_float* targets;
  gsl_matrix_float* test_sample;
  uint16_t sample_count;
  uint16_t target_count;
  uint16_t samples_count;
  uint16_t targets_count;
  uint16_t result_count;

 public:
  explicit Organizer(const Slfn &network);
  ~Organizer();
  void buildSample(float value, Mode mode);
  void buildTarget(float value);
  void storeSample(Mode mode);
  void setTarget(void);
  uint16_t getSampleCount(void);
  uint16_t getTargetCount(void);
  uint16_t getSamplesCount(void);
  uint16_t getTargetsCount(void);
  gsl_matrix_float* getSamples(void);
  gsl_matrix_float* getTargets(void);
  gsl_matrix_float* getTestSample(void);
  void resetSamplesCount(void);
  void resetResultCount(void);
  void setResult(float value);
  float getResult(int index);
};

#endif  // INC_ELM_ORGANIZER_HPP_
