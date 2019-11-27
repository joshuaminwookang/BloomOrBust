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

int main(int argc, char** argv) {
    
    if (argc != 3) {
        printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
        exit(1);
    }

    unsigned char *h_bloom_filter_array = (unsigned char*)calloc(M_NUM_BITS, sizeof(unsigned char));
    String *h_string_array = (String*)malloc(15 * sizeof(String));
    String *d_string_array;

    FILE *add_fp = fopen(argv[1], "r");
    FILE *check_fp = fopen(argv[2], "r");      
    
    fileToArray(add_fp, h_string_array);
    
    checkCudaErrors(cudaMalloc((void **) &d_string_array, 15*sizeof(String)));
    checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, 15*sizeof(String), cudaMemcpyHostToDevice));
    
    cudaFree(d_string_array);

    free(h_bloom_filter_array);
    free(h_string_array);

    return 0;
}
