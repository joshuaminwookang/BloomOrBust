/* 
 * Bloom filter with CUDA.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "bloom.h"

#define BLOCK_SIZE 64.0

__device__ unsigned long cuda_hashstring(char *word)
{
  unsigned char *str = (unsigned char *)word;
  unsigned long hash = HASH_NUM;

  while (*str)
    {
      hash = ((hash << 5) + hash) + *(str++);
    }

  return hash;
}


__device__ void cuda_hash(long *hashes, char *word)
{
  unsigned long x = cuda_hashstring(word);
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
  cuda_hash(hashes, word);

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
    String *h_string_array = (String*)malloc(INIT_WORDS * sizeof(String));
    for (int i = 0; i < INIT_WORDS; i++)
    {
      strcpy(h_string_array[i].word, "");
    }
    
    // device arrays
    unsigned char *d_bf_array;
    String *d_string_array;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
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
    int num_words_added = fileToArray(add_fp, &h_string_array);

    int misses;

    
    // allocate device arrays
    checkCudaErrors(cudaMalloc((void **) &d_string_array, num_words_added*sizeof(String)));
    checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, num_words_added*sizeof(String), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_bf_array, M_NUM_BITS*sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_bf_array, h_bf_array, M_NUM_BITS*sizeof(unsigned char), cudaMemcpyHostToDevice));

    // set dimensions of blocks and grid
    //dim3 dimGrid(ceil(INIT_WORDS/BLOCK_SIZE), 1, 1);
    //dim3 dimBlock(BLOCK_SIZE, 1, 1);

    checkCudaErrors(cudaEventRecord(start));
    
    addToBloom<<<ceil(num_words/BLOCK_SIZE), BLOCK_SIZE>>>((unsigned char*)d_bf_array, (String*)d_string_array);

    checkCudaErrors(cudaEventRecord(stop));
    
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    misses = countMissFromFile(check_fp, h_bf_array);

    printInfo(num_words_added, -1, milliseconds, -1, misses);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_bf_array);
    cudaFree(d_string_array);
    free(h_bf_array);
    free(h_string_array);

    return 0;
}
