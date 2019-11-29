/*
 * Bloom filter with CUDA.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "bloom.h"

#define M_NUM_BITS 1024

int main(int argc, char** argv) 
{
    
    if (argc != 3) 
    {
        printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
        exit(1);
    }

    // host arrays
    unsigned char *h_bf_array = (unsigned char*)calloc(M_NUM_BITS, sizeof(unsigned char));
    String *h_string_array = (String*)malloc(15 * sizeof(String));
    
    // device arrays
    unsigned char *d_bf_array;
    String *d_string_array;

    // open files
    FILE *add_fp = fopen(argv[1], "r");
    if (add_fp == NULL)
    {
        printf("Failed to open file1. \n");
        exit(1);
    }

    FILE *check_fp = fopen(argv[2], "r");
    if (check_fp == NULL)
    {
        printf("Failed to open file2. \n");
        exit(1);
    }    
    
    // read in file1
    fileToArray(add_fp, h_string_array);
    
    // allocate device arrays
    checkCudaErrors(cudaMalloc((void **) &d_string_array, 15*sizeof(String)));
    checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, 15*sizeof(String), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_bf_array, 15*sizeof(String)));
    checkCudaErrors(cudaMemcpy(d_bf_array, h_bf_array, 15*sizeof(String), cudaMemcpyHostToDevice));
    
    cudaFree(d_string_array);

    free(h_bloom_filter_array);
    free(h_string_array);

    return 0;
}
