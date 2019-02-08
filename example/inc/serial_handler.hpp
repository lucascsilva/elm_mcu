#ifndef __SERIAL_HANDLER_HPP__
#define __SERIAL_HANDLER_HPP__

#include <array>
#include "elm.hpp"
#include "mbed.h"


class SerialHandler : private Serial
{
    private:
    
    std::array<float,NUM_SAMPLES> buffer;
    std::array<uint32_t,4> float_bytes;
    uint16_t buffer_counter;
    uint8_t float_byte_counter; 

    public:

    SerialHandler(PinName tx, PinName rx);
    float* getBufferElement(uint16_t position);
    void serialISR(void);
    void mergeBytes(void);
};

#endif