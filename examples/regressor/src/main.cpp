#include "mbed.h"
#include "elm.hpp"
#include "data_converter.hpp"
#include "organizer.hpp"
#include "slfn.hpp"


int main(void) 
{   
    Serial uart(USBTX,USBRX,9600);
    const Slfn network{
        1,  //input nodes count
        10,  //hidden neurons count
        1, //hidden layers count
        1,  //output neurons count
        40, //training set count
        20, //test set count
    };
    Organizer set(&network);

    //samples
    {
        DataConverter float_converter;
        while(set.getSamplesCount() < network.training_set_count)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(), TRAIN, &network); 
            }
        }
    }
    //targets
    {
        DataConverter float_converter;
        while(set.getTargetsCount() < network.training_set_count)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());  
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildTarget(float_converter.getConvertedFloat(), &network);
            } 
        }
    }

    Elm elm_network(&network);
    //training
    elm_network.TrainElm(set.getSamples(),set.getTargets(), &network);
    //running

    OutputData output_data;
    DigitalOut led(LED4);
    output_data.output = gsl_matrix_alloc(1, network.output_neurons_count);
    set.resetSamplesCount();
    uint32_t test_counter=0;
    while( test_counter < network.test_set_count)
    {
        DataConverter float_converter;
        while(set.getSamplesCount() < 1)
        {
            if(uart.readable())
            {
                float_converter.addByte(uart.getc());         
                if(float_converter.getConversionStatus()==COMPLETE)
                    set.buildSample(float_converter.getConvertedFloat(),TEST, &network);
            } 
        }
        set.resetSamplesCount();
        elm_network.NetworkOutput(set.getTestSample(),output_data.output, &network);
        set.setResult((float) gsl_matrix_get(output_data.output,0,0));
        test_counter++;
    }
    while(1)
    {
        led=!led;
        wait(0.5);
    }
}