#include "mbed.h"
#include "matrix_manipulation.h"
#include "mems.hpp"

#define LINES   10
#define COLUMNS 20

int main(void) 
{   
    DigitalOut orange_led(LED3,0);
    DigitalOut green_led(LED4,0);
    DigitalOut red_led(LED5,0);
    DigitalOut blue_led(LED6,1);
    Serial pc(USBTX,USBRX,115200);

    Lis3dh mems;
    
    /*gsl_matrix *A = gsl_matrix_alloc (LINES, COLUMNS);
    gsl_matrix *A_inv=gsl_matrix_alloc (COLUMNS, LINES);

    double counter=1;
    for(size_t row_counter=0 ;row_counter < A->size1; row_counter++)
    {
        for(size_t col_counter=0;col_counter<A->size2; col_counter++)
        {
            gsl_matrix_set(A,row_counter, col_counter, counter++);
        }   
    }

	A_inv = moore_penrose_pinv(A, 1e-8);
            
	gsl_matrix_free(A);
	gsl_matrix_free(A_inv);*/
    blue_led=0;
    mems.write(CTRL_REG1, CTRL_REG1_CONFIG);

    while(1)
    {
        orange_led = !orange_led;
        mems.update();
        pc.putc((uint8_t)((mems.getX() & 0xFF00)>>8));
        pc.putc((uint8_t)(mems.getX() & 0x00FF));
        pc.putc('\n');
        pc.putc((uint8_t)((mems.getY() & 0xFF00)>>8));
        pc.putc((uint8_t)(mems.getY() & 0x00FF));
        pc.putc('\n');
        pc.putc((uint8_t)((mems.getZ() & 0xFF00)>>8));
        pc.putc((uint8_t)(mems.getZ() & 0x00FF));
        pc.putc('\n');
        pc.putc('\n');
        wait(0.5);
    }
}
