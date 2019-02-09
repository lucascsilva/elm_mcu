#ifndef __DATA_CONVERTER_HPP__
#define __DATA_CONVERTER_HPP__

#include <array>

typedef enum _ConversionStatus
{
    RECEIVING_BYTES=0,
    COMPLETE,
}ConversionStatus;

class DataConverter
{
    private:

    std::array<uint32_t,4> float_bytes;
    uint8_t float_byte_counter; 
    float float_conversion;
    ConversionStatus conversion_status; 

    public:

    void addByte(uint32_t byte);
    void floatConversion(void);
    float getConvertedFloat(void);
};

#endif