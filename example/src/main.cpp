#include "mbed.h"
#include "mems.hpp"
#include "elm.hpp"
#include "config.h"
#include "data_converter.hpp"
#include "organizer.hpp"


int main(void) 
{   
    Serial uart(USBTX,USBRX,9600);
    Organizer training_set_organizer;

    //samples
    {
        DataConverter float_converter;
        while(training_set_organizer.getSamplesCount()<NUM_SAMPLES)
        {
            if(uart.readable())
                float_converter.addByte(uart.getc());         
            if(float_converter.getConversionStatus()==COMPLETE)
                training_set_organizer.buildSample(float_converter.getConvertedFloat(),TRAIN); 
        }
    }
    //targets
    {
        DataConverter float_converter;
        while(training_set_organizer.getTargetsCount()<NUM_SAMPLES)
        {
            if(uart.readable())
                float_converter.addByte(uart.getc());         
            if(float_converter.getConversionStatus()==COMPLETE)
                training_set_organizer.buildTarget(float_converter.getConvertedFloat()); 
        }
    }

    Elm elm_network;
    //training
    elm_network.TrainElm(training_set_organizer.getSamples(),training_set_organizer.getTargets());

    //running
    while(1)
    {
        Organizer test_sample;
        DataConverter float_converter;
        OutputData output_data;
        while(test_sample.getSampleCount() < NUM_INPUT_NEURONS)
        {
            if(uart.readable())
                float_converter.addByte(uart.getc());         
            if(float_converter.getConversionStatus()==COMPLETE)
                test_sample.buildSample(float_converter.getConvertedFloat(),TEST); 
        }
        elm_network.NetworkOutput(test_sample.getTestSample(),output_data.output);
    }
}


    /*DigitalOut orange_led(LED3,0);
    DigitalOut green_led(LED4,0);
    DigitalOut red_led(LED5,0);
    DigitalOut blue_led(LED6,0);

    DigitalIn train_button(USER_BUTTON);
    DigitalIn train_orange(PB_11,PullDown);
    DigitalIn train_green(PB_12,PullDown);
    DigitalIn train_red(PB_13,PullDown);
    DigitalIn train_blue(PB_14,PullDown);

    Elm elm_network;
    Lis3dh mems;
    gsl_matrix *samples; 
    gsl_matrix *target;
    gsl_matrix *sample;
    
    OutputData output_data;

    uint8_t samples_count=0;

    if(mems.read(CTRL_REG1)!= CTRL_REG1_CONFIG)
        mems.write(CTRL_REG1, CTRL_REG1_CONFIG);
    
    samples = gsl_matrix_alloc(NUM_INPUT_NEURONS,NUM_SAMPLES);
    target = gsl_matrix_calloc(NUM_SAMPLES,NUM_OUTPUT_NEURONS);

    blue_led=1;
    while(!train_button);
    blue_led=0;
    
    pc.printf("Sampling started");
    wait(0.2);
    green_led=1;

    while(samples_count<NUM_SAMPLES)
    {
        while(!train_button);
        wait(0.2);
        green_led=0;
        orange_led=1;
        mems.update();
        gsl_matrix_set(samples, 0, samples_count,(double)mems.getX());
        gsl_matrix_set(samples, 1, samples_count,(double)mems.getY());
        gsl_matrix_set(samples, 2, samples_count,(double)mems.getZ());
        gsl_matrix_set(target, samples_count, 0, train_orange);
        gsl_matrix_set(target, samples_count, 1, train_green);
        //gsl_matrix_set(target, samples_count, 2, train_blue);
        //gsl_matrix_set(target, samples_count, 3, train_red);
        //gsl_matrix_set(target, samples_count, 4, !(train_orange|train_green|train_blue|train_red));//active if no button pressed
        samples_count++;
        orange_led=0;
        green_led=1;
    }

    green_led=0;

    //training
    red_led=1;
    pc.printf("Training started");
    elm_network.TrainElm(samples,target);
    pc.printf("Training finished");
    red_led=0;
    

    gsl_matrix_free(samples);
    gsl_matrix_free(target);

    //after training
    sample = gsl_matrix_alloc(NUM_INPUT_NEURONS,1);
    output_data.output = gsl_matrix_alloc(1,NUM_OUTPUT_NEURONS);

    while(1)
    {   
        wait(0.5);
        mems.update();    
        gsl_matrix_set(sample, 0, 0,(double)mems.getX());
        gsl_matrix_set(sample, 1, 0,(double)mems.getY());
        gsl_matrix_set(sample, 2, 0,(double)mems.getZ());
        elm_network.NetworkOutput(sample,output_data.output);
        gsl_matrix_max_index(output_data.output, &output_data.row_max, &output_data.col_max);
        switch(output_data.col_max)
        {
            case OUTPUT_LED_ORANGE:
                orange_led=1;
                green_led=0;
                blue_led=0;
                red_led=0;
                break;
            case OUTPUT_LED_GREEN:
                orange_led=0;
                green_led=1;
                blue_led=0;
                red_led=0;
                break;
            case OUTPUT_LED_BLUE:
                orange_led=0;
                green_led=0;
                blue_led=1;
                red_led=0;
                break;
            case OUTPUT_LED_RED:
                orange_led=0;
                green_led=0;
                blue_led=0;
                red_led=1;
                break;
            case OUTPUT_LED_ALL:
                orange_led=1;
                green_led=1;
                blue_led=1;
                red_led=1;
                break;
        }
    }*/
