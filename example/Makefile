# Copyright 2017 Adam Green (http://mbed.org/users/AdamGreen/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
PROJECT         := ELM
DEVICES         := DISCO_F407VG
VERBOSE 	    := 1
GCC4MBED_DIR    := ../../../gcc4mbed
GCC4MBED_TYPE 	:= Release
NO_FLOAT_SCANF  := 1
NO_FLOAT_PRINTF := 1
USER_LIBS	    := ../src ../inc
INCDIRS		    := inc ../inc
LIBS_PREFIX 	:= ../lib/libgsl.a ../lib/libgslcblas.a
LIBS_SUFFIX 	:= ../lib/libgsl.a ../lib/libgslcblas.a


include $(GCC4MBED_DIR)/build/gcc4mbed.mk

burn:
	openocd -f "stm32f4discovery.cfg" -c "program DISCO_F407VG/ELM.bin 0x8000000 verify reset exit"

