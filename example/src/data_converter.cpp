#include "data_converter.hpp"

void DataConverter::addByte(uint32_t byte)
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
    float_conversion=(float)(float_bytes[0] + (float_bytes[1]<<8) + (float_bytes[2]<<16) + (float_bytes[3]<<24));
}

float DataConverter::getConvertedFloat(void)
{
    if (conversion_status == COMPLETE);
        return float_conversion;
    else
        return NULL
}