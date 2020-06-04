# ELM API for microcontrollers

This bundle contains source files for a C++ implementation of Extreme Learning Machines (ELM) for microcontrollers. It is 
based on MbedOS and a GSL port for ARM Cortex-M4, which is provided as static library. 

Using this application, it is possible to train an artificial neural network in the single layer feedforward network (SLFN)
 topology following the ELM theory, and use it later once trained.
It leverages ELM in 2 ways:
- ELM is a super fast method for training artificial neural networks, therefore suitable for the processing power
of microcontrollers - no iterations of extensive calculations are needed.
- Since random weights need not to be tuned during the trained step, they can be stored in program memory and used for every
new training. 

There is though one difference in this application, which is random weight generation according to a binary enconding scheme, 
so that the random weights connecting the input layer to the hidden layer will have 2^b distinct values uniformly sorted within 
the interval [-1,1], where b is the bit count. For instance, selecting b=1 will lead to random weights -1 and 1 assigned in the
hidden laye. Selecting b=2 will provide a pool containing the values -1, -0.5, 0.5 and 1, from which the random weights will be 
assigned randomly, and so on for grater values of b.

This application was designed originally to work with ultrasonic signals from weld beads, on a STMF4 Discovery development board,
as part of a scientific research. Users are encouraged to employ it in scientific research, engineering and robotic projects and 
contribute to its development. In case of using this application in reasearch, please cite it:

Author: Lucas Cruz da Silva

e-mail: lucas.silva@ymail.com

Date: May 2020

# Installation

This application uses MbedOS, therefore it shall be installed on the working environment. MBED-CLI works fine. Once 
[installed](https://os.mbed.com/docs/mbed-os/v5.15/tools/installation-and-setup.html), initialize a new project in 
the workspace with

```mbed new [program_name]```

then put the files here inside a new folder (e.g. `elm/`) in the newly created `program_name` folder. The file tree should look like
```
|— program_name
   |— elm/
      |—inc/
      |—lib/
      |—src/
      |—examples/
   |—mbed-os/
```
Target and toolchain must also be [configured](https://os.mbed.com/docs/mbed-os/v5.15/tools/configuration-options.html). 
This application is run using DISCO_F407VG and GCC_ARM respectively (install the [ARM toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) in case it is missing.

The file `build_config.json` shall be invoked in compilation by using, referring to the above example, `mbed compile --app-config elm/build_config.json` as it contains the include paths and library path flags that are passed to the compiler.

# Usage

The application is based on 3 classes:
- `Slfn` holds the configuration of the neural netowrk, such as input node count, hidden neuron count, output neuron count, and so on.
- `Elm`contains the random weights, random bias and output weights of the network.
- `Organizer`is a handler for dealing with sample sets, containing the training set, targets, test sample and builders for multidimensional
samples.

The general procedure is to create an `Slfn` object and use it to manage the sample sets via `Organizer` object. Once the samples
are collected, finally run the training method of the `Elm` network. If the network was already trained, feed it with test samples
in order to obtain the network output.

# Documentation

Please run doxygen in order to obtain the code documentation. 

# Application example

An application example for a multiclass problem is provided in the path `examples/multiclass/`,
which was used in the aforementioned research. This example expects samples and targets in float format (4 bytes per value) whose size
is defined in an `Slfn` object, then trains the network, waits for test samples and calculate the outputs of the trained network.


