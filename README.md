# Bloom or Bust

(c) 2019 Josh Minwoo Kang and Andrew Thai

Hardware accelerator and CUDA programming for accelerating Bloom filter based information queries.

Note that this implementation emulates Bloom filter behavior.

## Important Files

bloom.h: Header file containing many useful functions for our
implementation along with helpful testing functions

bloom.c: Sequential Implementation.

bloom.cu: CUDA Implementation


## How to Compile

'make'


## How to Run

Sequential Implementation: './bloom-seq file1 file2'

CUDA Implementation: './bloom-cuda file1 file2'


## To Use CUDA Optimizations

- Open bloom.cu

- Uncomment the following to use optimizations:

#SHARED_MAP:  shared memory in map() kernel

#SHARED_TEST: shared memory in test() kernel

#SORT: 	      sort arrays before processing

- Edit BLOCK_SIZE to a float to try different block sizes.

## Information on the HW Accelerator Code

- hw_src/bloom.scala is the Chisel code to generate BF accelerator as RoCC co-processor
- hw_src/hw-bloom-script.c is the test script that uses BF RoCC instructions
- hw_src/sw-bloom-script.c is the test script that only uses software to compute the same Map() and Test() operations
- Comment/Uncomment to direct the test scripts to use different data sets:

#TINY  : use 11 word data for both map and test

#TINYV2  :  use 30 word data for both map and test

#TINYV3_Map  : use 50 word data for map and 30 word for test

#TINYV3_Test : use 50 word data for map and 50 word for test

- *_data.h are header fills containing hard-coded word arrays (we need this because the RISC-V gcc does not support file or stream-based library functions, nor dynamic memory allocation)

Chisel source files should be placed within the Rocket-chip directory: 

https://github.com/freechipsproject/rocket-chip

and as in our case, it is a good idea to use Berkeley's Chipyard framework: 

https://github.com/ucb-bar/chipyard

To generate RISC-V binary workloads from our test scripts, please refer to the FireMarshal Documentation:
https://github.com/firesim/FireMarshal
