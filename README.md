# BloomOrBust

(c) 2019 Josh Minwoo Kang and Andrew Thai

Hardware accelerator and CUDA programming for accelerating Bloom filter based information queries.
This implementation emulates Bloom filter behavior.

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
