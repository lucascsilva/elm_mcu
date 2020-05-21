/*
* Copyright 2019
* Lucas Silva
*/
#ifndef INC_UTILS_DATA_CONVERTER_HPP_
#define INC_UTILS_DATA_CONVERTER_HPP_

#include <array>
#include <cstring>


/**
 * @enum data_converter.hpp "inc/utils/data_converter.hpp"
 * 
 * @brief Status for conversion state machine
 */
enum ConversionStatus {
  RECEIVING_BYTES = 0,  /**< Receiving bytes (up to four) to assemble a float variable*/
  COMPLETE              /**< 4 bytes have been received to form a float variable*/
};


/**
 * @class data_converter.hpp "inc/utils/data_converter.hpp"
 * 
 * @brief Handler for retrieving float values from bytes. It was meant to be used in 
 * serial communications, where data is transferred bytewise.
 */
class DataConverter {
 private:
  /** Array of component bytes of a float variable*/
  std::array<uint8_t, 4> float_bytes;
  /** Counter used in float building*/
  uint8_t float_byte_counter;
  /**Stores the conversion from bytes to float*/
  float float_conversion;
  /** Integer containing the merged bytes*/
  uint32_t merged_bytes;
  /*Stores the current state of float building*/
  ConversionStatus conversion_status;
  /**
   * @brief Converts the 4-byte compound into a float variable and stores is in DataConverter::float_conversion
   */
  void floatConversion(void);

 public:
  /** 
  * @brief Constructor
  */
  DataConverter();
  /** 
   * @brief Adds a byte to the float byte array
   * 
   * @param byte new by to be added to the float byte array
   */
  void addByte(uint8_t byte);
  /**
   * @brief Returns the currently stored float value
   */
  float getConvertedFloat(void);
  /**
   * @brief Returns the current state in float conversion
   */
  ConversionStatus getConversionStatus(void);
};

#endif  // INC_UTILS_DATA_CONVERTER_HPP_
