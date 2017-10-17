#include "mbed.h"
#include "matrix_manipulation.h"
#include "mems.hpp"
#include "elm.hpp"
#include "config.h"


int main(void) 
{   
    DigitalOut orange_led(LED3,0);
    DigitalOut green_led(LED4,0);
    DigitalOut red_led(LED5,0);
    DigitalOut blue_led(LED6,0);

    DigitalIn train_button(USER_BUTTON);
    DigitalIn train_orange(PE_0,PullDown);
    DigitalIn train_green(PE_1,PullDown);
    DigitalIn train_red(PE_2,PullDown);
    DigitalIn train_blue(PE_3,PullDown);
    
    Elm elm_network;
    Lis3dh mems;
    gsl_matrix *samples; 
    gsl_matrix *target;
    gsl_matrix *sample;
    
    OutputData output_data;

    uint8_t samples_count=0;
    
    mems.write(CTRL_REG1, CTRL_REG1_CONFIG);
    samples = gsl_matrix_alloc(NUM_INPUT_NEURONS,NUM_SAMPLES);
    target = gsl_matrix_calloc(NUM_SAMPLES,NUM_OUTPUT_NEURONS);


    while(!train_button);
    
    green_led=1;

    while(samples_count<NUM_SAMPLES)
    {
        while(!train_button);
        green_led=0;
        orange_led=1;
        mems.update();
        gsl_matrix_set(samples, 0, NUM_SAMPLES,(double)mems.getX());
        gsl_matrix_set(samples, 1, NUM_SAMPLES,(double)mems.getY());
        gsl_matrix_set(samples, 2, NUM_SAMPLES,(double)mems.getZ());
        gsl_matrix_set(target, NUM_SAMPLES, 0, train_orange);
        gsl_matrix_set(target, NUM_SAMPLES, 1, train_green);
        gsl_matrix_set(target, NUM_SAMPLES, 2, train_blue);
        gsl_matrix_set(target, NUM_SAMPLES, 3, train_red);
        gsl_matrix_set(target, NUM_SAMPLES, 4, !(train_orange|train_green|train_blue|train_red));
        samples_count++;
        orange_led=0;
        green_led=1;
    }

    green_led=0;

    //training
    red_led=1;
    elm_network.TrainElm(samples,target);
    red_led=0;

    gsl_matrix_free(samples);
    gsl_matrix_free(target);

    //after training
    sample = gsl_matrix_alloc(NUM_INPUT_NEURONS,1);
    output_data.output = gsl_matrix_alloc(1,NUM_OUTPUT_NEURONS);

    while(1)
    {   
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
    }
}
