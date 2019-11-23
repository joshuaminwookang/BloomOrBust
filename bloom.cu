/*
 * Bloom filter with CUDA.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "bloom.h"



int main(int argc, char** argv) {
    
    if (argc != 3) {
        printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
        exit(1);
    }

    unsigned char *h_bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));
    String *h_string_array = (String*)malloc(15 * sizeof(String));
    

    FILE *add_fp = fopen(argv[1], "r");
    FILE *check_fp = fopen(argv[2], "r");      
    
    fileToArray(add_fp, string_array);

    free(bloom_filter_array);
    free(string_array);

    return 0;
}