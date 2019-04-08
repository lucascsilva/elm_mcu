#include "data_converter.hpp"

DataConverter::DataConverter()
    :float_byte_counter(0),
    conversion_status(RECEIVING_BYTES) 
{
    
}

void DataConverter::addByte(uint8_t byte)
{
    switch(conversion_status)
    {
        case RECEIVING_BYTES:
            float_bytes[float_byte_counter++]= byte;
            if(float_byte_counter == float_bytes.size())
            {
                conversion_status = COMPLETE;
                floatConversion();
            }
            break;
        case COMPLETE:
            float_byte_counter=0;
            float_bytes[float_byte_counter++]= byte;
            conversion_status = RECEIVING_BYTES; 
            break;
    }  
} 

void DataConverter::floatConversion(void)
{   
    std::memcpy(&float_conversion, &float_bytes, float_bytes.size());
}

float DataConverter::getConvertedFloat(void)
{
        return float_conversion;
}

ConversionStatus DataConverter::getConversionStatus(void)
{
    return conversion_status;
}