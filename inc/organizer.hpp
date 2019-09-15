/**
 * Copyright 2019
 * @file organizer.hpp
 * @author Lucas Silva
 */

#ifndef INC_ORGANIZER_HPP_
#define INC_ORGANIZER_HPP_

#include "gsl/gsl_blas.h"
#include "gsl/gsl_linalg.h"
#include "slfn.hpp"

enum Mode {
  TRAIN = 0,
  TEST
};

struct OutputData {
  gsl_matrix *output;
  size_t row_max;
  size_t col_max;
};

class Organizer {
 private:
  float* sample;
  float* target;
  float* result;
  gsl_matrix* training_set;
  gsl_matrix* targets;
  gsl_matrix* test_sample;
  uint16_t sample_count;
  uint16_t target_count;
  uint16_t samples_count;
  uint16_t targets_count;
  uint16_t result_count;

 public:
  Organizer(const Slfn* network);
  ~Organizer();
  void buildSample(float value, Mode mode, const Slfn* network);
  void buildTarget(float value, const Slfn* network);
  void storeSample(Mode mode, const Slfn* network);
  void setTarget(const Slfn* network);
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

#endif  // INC_ORGANIZER_HPP_
