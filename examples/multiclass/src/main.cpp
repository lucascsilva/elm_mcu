/*
* Copyright 2019
*  Lucas Silva
*/
#include "mbed.h"
#include "../inc/elm/elm.hpp"
#include "../inc/elm/organizer.hpp"
#include "../inc/utils/data_converter.hpp"


int main(void) {
  Serial uart(USBTX, USBRX, 115200);
  Slfn parameters;
  parameters.input_nodes_count = 10;
  parameters.hidden_neurons_count = 20;
  parameters.hidden_layers_count = 1;
  parameters.output_neurons_count = 5;
  parameters.training_set_count = 700;
  parameters.test_set_count = 300;
  parameters.output_neuron_type = ADDITIVE;
  Organizer set(parameters);

  // samples
  {
    DataConverter float_converter;
    while (set.getSamplesCount() < parameters.training_set_count) {
      if (uart.readable()) {
        float_converter.addByte(uart.getc());
        if (float_converter.getConversionStatus() == COMPLETE)
          set.buildSample(float_converter.getConvertedFloat(), TRAIN);
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

  Elm elm_network(parameters);
  // training
  elm_network.TrainElm(set.getSamples(), set.getTargets());
  // running
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
          set.buildSample(float_converter.getConvertedFloat(), TEST);
      }
    }
    set.resetSamplesCount();
    elm_network.NetworkOutput(set.getTestSample(), output_data.output);
    for (size_t result_counter = 0; result_counter < parameters.output_neurons_count; result_counter++) {
      set.setResult(gsl_matrix_float_get(output_data.output, 0, result_counter));
    }
    test_counter++;
  }
  while (true) {
    led = !led;
    wait(0.5);
  }
}
