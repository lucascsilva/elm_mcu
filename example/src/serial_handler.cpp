#include "elm.hpp"
#include "mbed.h"
#include "serial_handler.hpp"

SerialHandler::SerialHandler(PinName tx, PinName rx)
    :Serial(tx,rx)
{
    buffer_counter=0;
    float_byte_counter=0;
    Serial::attach(callback(this, &SerialHandler::serialISR));
}

float* SerialHandler::getBufferElement(uint16_t position)
{
    return buffer.data()+position;
}

void SerialHandler::serialISR(void)
{
    float_bytes[float_byte_counter++]=Serial::getc();
    if(float_byte_counter == float_bytes.size())
    {
        float_byte_counter=0;
        mergeBytes();
    }   
}

void SerialHandler::mergeBytes(void)
{   
    buffer[buffer_counter++]=(float)(float_bytes[0] + (float_bytes[1]<<8) + (float_bytes[2]<<16) + (float_bytes[3]<<24));
    if (buffer_counter == float_bytes.size())
        buffer_counter=0;
}