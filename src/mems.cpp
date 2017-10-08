#include "mems.hpp"

SPI spi_mems(PA_7, PA_6, PA_5);
DigitalOut cs(PE_3,1);

void Lis3dh::update(void)
{
    x=(uint16_t)read(OUT_X_L);
    x|=(uint16_t)(read(OUT_X_H)<<8);
    x=x>>6; //10 bit output

    y=(uint16_t)read(OUT_Y_L);
    y|=(uint16_t)(read(OUT_Y_H)<<8);
    y=y>>6; //10 bit output
    
    z=(uint16_t)read(OUT_Z_L);
    z|=(uint16_t)(read(OUT_Z_H)<<8);
    z=z>>6; //10 bit output
} 

uint8_t Lis3dh::read(uint8_t address)
{ 
    uint8_t data;
    address|=0x80;
    
    cs=0;
    spi_mems.write(address);
    data=spi_mems.write(0x00);
    cs=1;
    return data;
}

void Lis3dh::write(uint8_t address, uint8_t byte)
{  
    cs=0;
    spi_mems.write(address);
    spi_mems.write(byte);
    cs=1;
}

uint16_t Lis3dh::getX(void)
{
    return x;
}

uint16_t Lis3dh::getY(void)
{
    return y;
}

uint16_t Lis3dh::getZ(void)
{
    return z;
}


