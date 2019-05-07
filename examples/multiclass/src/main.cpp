#include "mbed.h"
#include "elm.hpp"
#include "data_converter.hpp"
#include "organizer.hpp"
#include "slfn.hpp"


int main(void) 
{   
    Serial uart(USBTX,USBRX,9600);
    const Slfn parameters{
        1,  //input nodes count
        10,  //hidden neurons count
        1, //hidden layers count
        2,  //output neurons count
        40, //training set count
        20, //test set count
    };
    Organizer set(&parameters);

    //samples
    {
        DataConverter float_converter;
        while(set.getSamplesCount() < parameters.training_set_count)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(), TRAIN, &parameters); 
            }
        }
    }
    //targets
    {
        DataConverter float_converter;
        while(set.getTargetsCount() < parameters.training_set_count)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());  
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildTarget(float_converter.getConvertedFloat(), &parameters);
            } 
        }
    }

    Elm elm_network(&parameters);
    //training
    elm_network.TrainElm(set.getSamples(),set.getTargets(), &parameters);
    //running

    OutputData output_data;
    DigitalOut led(LED4);
    output_data.output = gsl_matrix_alloc(1, parameters.output_neurons_count);
    set.resetSamplesCount();
    uint32_t test_counter=0;
    while( test_counter < parameters.test_set_count)
    {
        DataConverter float_converter;
        while(set.getSamplesCount() < 1)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(),TEST, &parameters);
            } 
        }
        set.resetSamplesCount();
        elm_network.NetworkOutput(set.getTestSample(),output_data.output, &parameters);
        set.setResult((float) gsl_matrix_get(output_data.output,0,0));
        set.setResult((float) gsl_matrix_get(output_data.output,0,1));
        test_counter++;
    }
    while(1)
    {
        led=!led;
        wait(0.5);
    }
}