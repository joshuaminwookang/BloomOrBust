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

/* 
 * Hash String in CUDA.
 */
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

/*
 * Hash string to multiple indices in CUDA.
 */
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

__device__ int cuda_checkBloom(unsigned char *filter, char *word)
{
  long hashes[K_NUM_HASH];
  cuda_hash(hashes, word);

  for (int i = 0; i < K_NUM_HASH; i++)
  {
    // miss
    if (!filter[hashes[i]])
    {
      return 1; // +1 for a miss
    }
  }

  return 0;
}

__global__ void cuda_countMisses(unsigned char *filter, String *words, int *count, int num_words)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_words)
  {
    int miss = cuda_checkBloom(filter, words[index].word);
    atomicAdd(count, miss);
  }
}

/*
 * Set bits in Bloom filter based on hash values.
 */
__device__ void cuda_mapToBloom(unsigned char *filter, char *word)
{
  long hashes[K_NUM_HASH];
  cuda_hash(hashes, word);

  for (int i = 0; i < K_NUM_HASH; i++)
  {
    filter[hashes[i]] = 1;
  }
}

__global__ void cuda_addToBloom(unsigned char *bf_array, String *words, int num_words)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_words)
  {
    cuda_mapToBloom(bf_array, words[index].word);
  }
}

int main(int argc, char **argv)
{

  if (argc != 3)
  {
    printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
    exit(1);
  }

  // host data
  unsigned char *h_bf_array = (unsigned char *)calloc(M_NUM_BITS, sizeof(unsigned char));
  String *h_string_array = (String *)malloc(INIT_WORDS * sizeof(String));
  String *h_check_array = (String *)malloc(INIT_WORDS * sizeof(String));
  int h_misses[1];
  h_misses[0] = 0;

  for (int i = 0; i < INIT_WORDS; i++)
  {
    strcpy(h_string_array[i].word, "");
  }

  for (int i = 0; i < INIT_WORDS; i++)
  {
    strcpy(h_check_array[i].word, "");
  }

  // device data
  unsigned char *d_bf_array;
  String *d_string_array;
  String *d_check_array;
  int *d_misses;

  // time measurement
  float add_time, check_time = 0;
  cudaEvent_t start_add, stop_add, start_check, stop_check;
  checkCudaErrors(cudaEventCreate(&start_add));
  checkCudaErrors(cudaEventCreate(&stop_add));
  checkCudaErrors(cudaEventCreate(&start_check));
  checkCudaErrors(cudaEventCreate(&stop_check));

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
  int num_words_check = fileToArray(check_fp, &h_check_array);

  // allocate device arrays
  checkCudaErrors(cudaMalloc((void **)&d_string_array, num_words_added * sizeof(String)));
  checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, num_words_added * sizeof(String), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_check_array, num_words_check * sizeof(String)));
  checkCudaErrors(cudaMemcpy(d_check_array, h_check_array, num_words_check * sizeof(String), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_bf_array, M_NUM_BITS * sizeof(unsigned char)));
  checkCudaErrors(cudaMemcpy(d_bf_array, h_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyHostToDevice));

  // add words to Bloom filter
  checkCudaErrors(cudaEventRecord(start_add));
  cuda_addToBloom<<<ceil(num_words_added / BLOCK_SIZE), BLOCK_SIZE>>>((unsigned char *)d_bf_array,
                                                                      (String *)d_string_array,
                                                                      num_words_added);
  checkCudaErrors(cudaEventRecord(stop_add));

  // get running time
  checkCudaErrors(cudaEventSynchronize(stop_add));
  checkCudaErrors(cudaEventElapsedTime(&add_time, start_add, stop_add));

  checkCudaErrors(cudaMalloc((void **)&d_misses, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_misses, h_misses, sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventRecord(start_check));
  cuda_countMisses<<<ceil(num_words_check / BLOCK_SIZE), BLOCK_SIZE>>>((unsigned char *)d_bf_array,
                                                                       (String *)d_check_array,
                                                                       (int *)d_misses, num_words_check);
  checkCudaErrors(cudaEventRecord(stop_check));
  checkCudaErrors(cudaEventSynchronize(stop_check));
  checkCudaErrors(cudaEventElapsedTime(&check_time, start_check, stop_check));

  // get resulting Bloom filter
  checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_misses, d_misses, sizeof(int), cudaMemcpyDeviceToHost));

  int miss = countMissFromFile(check_fp, h_bf_array);

  // print run time info
  printInfo(num_words_added, num_words_check, add_time, check_time, *h_misses);

  // cleanup
  cudaEventDestroy(start_add);
  cudaEventDestroy(stop_add);
  cudaEventDestroy(start_check);
  cudaEventDestroy(stop_check);
  cudaFree(d_bf_array);
  cudaFree(d_string_array);
  free(h_bf_array);
  free(h_string_array);

  return 0;
}
