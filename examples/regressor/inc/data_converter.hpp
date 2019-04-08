#ifndef __DATA_CONVERTER_HPP__
#define __DATA_CONVERTER_HPP__

#include <array>
#include <cstring>

typedef enum _ConversionStatus
{
    RECEIVING_BYTES=0,
    COMPLETE,
}ConversionStatus;

class DataConverter
{
    private:

    std::array<uint8_t,4> float_bytes;
    uint8_t float_byte_counter; 
    float float_conversion;
    uint32_t merged_bytes;
    ConversionStatus conversion_status; 

    public:

    DataConverter();
    void addByte(uint8_t byte);
    void floatConversion(void);
    float getConvertedFloat(void);
    ConversionStatus getConversionStatus(void);
};

#endif