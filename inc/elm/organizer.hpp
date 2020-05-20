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

namespace elm {

/**
 * @enum Mode organizer.hpp "inc/elm/organizer.hpp"
 * 
 * @brief Defines the current network mode, either train or test.
 */ 
enum Mode {
  TRAIN = 0, /**< Training step*/
  TEST       /**< Testing*/
};

/**
 * @struct OutputData organizer.hpp "inc/elm/organizer.hpp"
 * 
 * @brief Support struct for viewing outputs of trained networks.
 */ 
struct OutputData {
  /** gsl_matrix containing output values of a trained netowrk. Must be initialized manually.*/
  gsl_matrix_float *output;
  /** Support variable for storing the row index of maximum value*/
  size_t row_max;
  /** Support variable for storing the column index of maximum value*/
  size_t col_max;
};

/**
 * @class Organizer organizer.hpp "inc/elm/organizer.hpp"
 * 
 * @brief Support class for Elm, meant to handle sample sets (training and test), targets
 * and the trained network's results
 */
class Organizer {
 private:
  /** Stores the network configuration*/
  Slfn network_config_;
  /**Array of values comprising a sample of size network_config_.input_nodes_count.*/
  float* sample;
  /**Array of values comprising a target of size network_config_.output_neurons_count.*/
  float* target;
  /** Array of results whose size is test set size times output neurons count.*/
  float* result;
  /**Stores the training set*/
  gsl_matrix_float* training_set;
  /** Stores the targets for the training set*/
  gsl_matrix_float* targets;
  /** Stores a test sample*/
  gsl_matrix_float* test_sample;
  /** Counter used to build a single sample in a float vector*/
  uint16_t sample_count;
  /**Counter used to build a single target in a float vector*/
  uint16_t target_count;
  /**Counter used to build gsl_matrix samples */
  uint16_t samples_count;
  /**Counter used to build gsl_matrix targets*/
  uint16_t targets_count;
  /**Counter used in float result*/
  uint16_t result_count;

 public:
   /**
   * @brief Constructor
   * 
   * Allocates memory blocks and initialize all counters to 0.
   * 
   * @param network Slfn structure containing the network configuration.
   */
  explicit Organizer(const Slfn &network);
  /**
  * @brief Default destructor
  * 
  * Deallocates memory blocks used by private the members
  */
  ~Organizer();
  /**
   * @brief Inputs a new value to the float sample vector and increments Organizer::sample_count. If it reaches
   * the input dimensionality (given by the input node count), it stores the sample in the corresponding 
   * gsl_matrix, either Organizer::training_set or Organizer::test_sample.
   * 
   * @param value The value to be stored in the current sample.
   * @param Mode Defines whether the sample must be stored in Organizer::training_set (Mode::TRAIN) or
   * in Organizer::test_sample (Mode::TEST).
   */ 
  void buildSample(float value, Mode mode);
  /**
   * @brief Inputs a new value to the float target vector and increments Organizer::target_count. If it reaches
   * the output dimensionality (given by the output node count), it stores the target in the corresponding
   * Organizer::targets.
   * 
   * @param value The value to be stored in the current target
   */
  void buildTarget(float value);
  /**
   * @brief Stores a sample in a gsl_matrix.
   * 
   * @param Mode Defines whether the sample must be stored in Organizer::training_set (Mode::TRAIN) or
   * in Organizer::test_sample (Mode::TEST).
   */
  void storeSample(Mode mode);
  /**
   * @brief Stores a target sample in Organizer::targets.
   */
  void setTarget(void);
  /**
   * @brief Returns the current value of Organizer:sample_count.
   * 
   * @returns Organizer::sample_count;
   */
  uint16_t getSampleCount(void);
  /**
   * @brief Returns the current value of Organizer:target_count.
   * 
   * @returns Organizer::target_count;
   */
  uint16_t getTargetCount(void);
  /**
   * @brief Returns the current value of Organizer:samples_count.
   * 
   * @returns Organizer::samples_count;
   */
  uint16_t getSamplesCount(void);
  /**
   * @brief Returns the current value of Organizer::targets_count.
   * 
   * @returns Organizer::targets_count;
   */
  uint16_t getTargetsCount(void);
  /**
   * @brief Returns the training set.
   * 
   * @returns Organizer::training_set;
   */
  gsl_matrix_float* getSamples(void);
  /**
   * @brief Returns the targets matrix.
   * 
   * @returns Organizer::targets;
   */
  gsl_matrix_float* getTargets(void);
  /**
   * @brief Returns the test sample.
   * 
   * @returns Organizer::test_sample;
   */
  gsl_matrix_float* getTestSample(void);
  /**
   * @brief resets the training_set count (Organizer::samples_count=0).
   */
  void resetSamplesCount(void);
  /**
   * @brief resets the results vector count (Organizer::result_count=0).
   */
  void resetResultCount(void);
  /**
   * @brief Stores a new value in Organizer::result and increments Organizer::result_count.
   * 
   * @param value The value to be stored.
   */ 
  void setResult(float value);
  /**
   * @brief Returns the value of Organizer::result in a given position.
   * 
   * @param index The position of the requested value in Organizer::result.
   * 
   * @returns The value stored in Organizer::result at position index.
   */
  float getResult(int index);
};
}  // namespace elm
#endif  // INC_ELM_ORGANIZER_HPP_
