/*
 * Bloom filter with CUDA.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "bloom.h"

__device__ unsigned long hashstring(char *word)
{
  unsigned char *str = (unsigned char *)word;
  unsigned long hash = HASH_NUM;

  while (*str)
    {
      hash = ((hash << 5) + hash) + *(str++);
    }

  return hash;
}


__device__ void hash(long *hashes, char *word)
{
  unsigned long x = hashstring(word);
  unsigned long y = x >> 4;

  for (int i = 0; i < K_NUM_HASH; i++)
    {
      x = (x + y) % M_NUM_BITS;
      y = (y + i) % M_NUM_BITS;
      hashes[i] = x;
    }
}

__device__ void mapToBloom(unsigned char *filter, char *word)
{
  long hashes[K_NUM_HASH];
  hash(hashes, word);

  for (int i = 0; i < K_NUM_HASH; i++)
    {
      filter[hashes[i]] = 1;
    }
}

__global__ void addToBloom(unsigned char *bf_array, String *words) 
{
  int index =  blockIdx.x * blockDim.x + threadIdx.x;
  mapToBloom(bf_array, words[index].word);
}

int main(int argc, char** argv) 
{
    
    if (argc != 3) 
    {
        printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
        exit(1);
    }

    // host arrays
    unsigned char *h_bf_array = (unsigned char*)calloc(M_NUM_BITS, sizeof(unsigned char));
    String *h_string_array = (String*)malloc(MAX_WORDS * sizeof(String));
    for (int i = 0; i < MAX_WORDS; i++)
    {
      strcpy(h_string_array[i].word, "");
    }
    
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
    checkCudaErrors(cudaMalloc((void **) &d_string_array, MAX_WORDS*sizeof(String)));
    checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, MAX_WORDS*sizeof(String), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_bf_array, M_NUM_BITS*sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_bf_array, h_bf_array, M_NUM_BITS*sizeof(unsigned char), cudaMemcpyHostToDevice));

    // set dimensions of blocks and grid
    //dim3 dimGrid(ceil(MAX_WORDS/32), 1, 1);
    //dim3 dimBlock(32, 1, 1);
    
    addToBloom<<<16, 32>>>((unsigned char*)d_bf_array, (String*)d_string_array);

    checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < M_NUM_BITS; i++) {
      printf("%d\n", h_bf_array[i]);
    }
    
    cudaFree(d_bf_array);
    cudaFree(d_string_array);

    free(h_bf_array);
    free(h_string_array);

    return 0;
}
