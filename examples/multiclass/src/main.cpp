/*
* Copyright 2019
*  Lucas Silva
*/
#include "mbed.h"
#include "elm/elm.hpp"
#include "elm/organizer.hpp"
#include "elm/slfn.hpp"
#include "utils/data_converter.hpp"


int main(void) {
  Serial uart(USBTX, USBRX, 9600);
  Slfn parameters;
  parameters.input_nodes_count = 1;
  parameters.hidden_neurons_count = 10;
  parameters.hidden_layers_count = 1;
  parameters.output_neurons_count = 2;
  parameters.training_set_count = 40;
  parameters.test_set_count = 20;
  parameters.output_neuron_type = ADDITIVE;
  Organizer set(&parameters);

  // samples
  {
    DataConverter float_converter;
    while (set.getSamplesCount() < parameters.training_set_count) {
      if (uart.readable()) {
        float_converter.addByte(uart.getc());
        if (float_converter.getConversionStatus() == COMPLETE)
          set.buildSample(float_converter.getConvertedFloat(), TRAIN, &parameters);
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
          set.buildTarget(float_converter.getConvertedFloat(), &parameters);
      }
    }
  }

  Elm elm_network(&parameters);
  // training
  elm_network.TrainElm(set.getSamples(), set.getTargets(), &parameters);
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
          set.buildSample(float_converter.getConvertedFloat(), TEST, &parameters);
      }
    }
    set.resetSamplesCount();
    elm_network.NetworkOutput(set.getTestSample(), output_data.output, &parameters);
    for (size_t result_counter = 0; result_counter < parameters.output_neurons_count; result_counter++) {
      set.setResult(gsl_matrix_float_get(output_data.output, 0, result_counter));
    }
    test_counter++;
  }
  while (1) {
    led = !led;
    wait(0.5);
  }
}
