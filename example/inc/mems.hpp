#ifndef __MEMS_HPP__
#define __MEMS_HPP__

#include "mbed.h"


#define OUT_X_L     0x28
#define OUT_X_H     0x29
#define OUT_Y_L     0x2A
#define OUT_Y_H     0x2B
#define OUT_Z_L     0x2C
#define OUT_Z_H     0x2D

#define WHO_AM_I    0x0F
#define CTRL_REG1   0x20

#define CTRL_REG1_CONFIG 0x27

class Lis3dh
{
    private:
    
    uint16_t x;
    uint16_t y;
    uint16_t z;

    public:

    void update(void);
    uint8_t read(uint8_t address); 
    void write(uint8_t address, uint8_t byte);

    uint16_t getX(void);
    uint16_t getY(void);
    uint16_t getZ(void);   
};

#endif