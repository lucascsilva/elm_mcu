/*
* Copyright 2019
* Lucas Silva
*/
#ifndef INC_UTILS_DATA_CONVERTER_HPP_
#define INC_UTILS_DATA_CONVERTER_HPP_

#include <array>
#include <cstring>

enum ConversionStatus {
  RECEIVING_BYTES = 0,
  COMPLETE
};

class DataConverter {
 private:
  std::array<uint8_t, 4> float_bytes;
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

#endif  // INC_UTILS_DATA_CONVERTER_HPP_
