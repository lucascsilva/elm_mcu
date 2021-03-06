/*
* Copyright 2019
*  Lucas Silva
*/
#include "mbed.h"
#include "../inc/elm/elm.hpp"
#include "../inc/elm/organizer.hpp"
#include "../inc/utils/data_converter.hpp"

using elm::Elm;
using elm::Organizer;
using elm::Slfn;
using elm::OutputData;
using elm::Mode;

int main(void) {
  Serial uart(USBTX, USBRX, 115200);
  Slfn parameters;
  parameters.input_nodes_count = 10;
  parameters.hidden_neurons_count = 20;
  parameters.hidden_layers_count = 1;
  parameters.output_neurons_count = 5;
  parameters.training_set_count = 700;
  parameters.test_set_count = 300;
  parameters.bits = 1;
  Organizer set(parameters);

  // samples
  {
    DataConverter float_converter;
    while (set.getSamplesCount() < parameters.training_set_count) {
      if (uart.readable()) {
        float_converter.addByte(uart.getc());
        if (float_converter.getConversionStatus() == COMPLETE)
          set.buildSample(float_converter.getConvertedFloat(), Mode::TRAIN);
      }
    }
  }
  // targets
  {
    DataConverter float_converter;
    while (set.getTargetsCount() < parameters.training_set_count) {
      if (uart.readable()) {
        float_converter.addByte(uart.getc());
        if (float_converter.getConversionStatus() == COMPLETE)
          set.buildTarget(float_converter.getConvertedFloat());
      }
    }
  }

  // training step
  Elm elm_network(parameters);
  elm_network.TrainElm(set.getSamples(), set.getTargets());
  // run the program and wait for test data
  OutputData output_data;
  DigitalOut led(LED4);
  output_data.output = gsl_matrix_float_alloc(1, parameters.output_neurons_count);
  set.resetSamplesCount();
  uint32_t test_counter = 0;
  while (test_counter < parameters.test_set_count) {
    DataConverter float_converter;
    while (set.getSamplesCount() < 1) {
      if (uart.readable()) {
        float_converter.addByte(uart.getc());
        if (float_converter.getConversionStatus() == COMPLETE)
          set.buildSample(float_converter.getConvertedFloat(), Mode::TEST);
      }
    }
    set.resetSamplesCount();
    elm_network.NetworkOutput(set.getTestSample(), output_data.output);
    for (size_t result_counter = 0; result_counter < parameters.output_neurons_count; result_counter++) {
      set.setResult(gsl_matrix_float_get(output_data.output, 0, result_counter));
    }
    test_counter++;
  }
  // Wait state. Debugging tools such as GDB may be used here to evaluate the results.
  while (true) {
    led = !led;
    wait(0.5);
  }
}
