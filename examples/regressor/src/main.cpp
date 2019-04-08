#include "mbed.h"
#include "elm.hpp"
#include "data_converter.hpp"
#include "organizer.hpp"


int main(void) 
{   
    Serial uart(USBTX,USBRX,9600);
    Organizer set;

    //samples
    {
        DataConverter float_converter;
        while(set.getSamplesCount()<NUM_SAMPLES)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(),TRAIN); 
            }
        }
    }
    //targets
    {
        DataConverter float_converter;
        while(set.getTargetsCount()<NUM_SAMPLES)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());  
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildTarget(float_converter.getConvertedFloat());
            } 
        }
    }

    Elm elm_network;
    //training
    elm_network.TrainElm(set.getSamples(),set.getTargets());
    //running

    OutputData output_data;
    DigitalOut led(LED4);
    output_data.output = gsl_matrix_alloc(1,NUM_OUTPUT_NEURONS);
    set.resetSamplesCount();
    int test_counter=0;
    while( test_counter< NUM_TEST)
    {
        DataConverter float_converter;
        while(set.getSamplesCount() < 1)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(),TEST);
            } 
        }
        set.resetSamplesCount();
        elm_network.NetworkOutput(set.getTestSample(),output_data.output);
        set.setResult((float) gsl_matrix_get(output_data.output,0,0));
        test_counter++;
    }
    while(1)
    {
        led=!led;
        wait(0.5);
    }
}