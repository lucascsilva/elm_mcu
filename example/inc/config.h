#ifndef __CONFIG_H__
#define __CONFIG_H__

typedef struct _OutputData
{
    gsl_matrix *output;
    size_t row_max;
    size_t col_max;
}OutputData;

typedef enum _OutputLed
{
    OUTPUT_LED_ORANGE=0,
    OUTPUT_LED_GREEN,
    OUTPUT_LED_RED,
    OUTPUT_LED_BLUE,
    OUTPUT_LED_ALL
}OutputLEd;

#endif