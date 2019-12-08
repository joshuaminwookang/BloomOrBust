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

__device__ int cuda_testBloom(unsigned char *filter, char *word)
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
    int miss = cuda_testBloom(filter, words[index].word);
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

__global__ void cuda_mapToBloom(unsigned char *bf_array, String *words, int num_words)
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
    printf("Usage: ./bloom WordsToMap WordsTotest\n");
    exit(1);
  }

  // host data
  unsigned char *h_bf_array = (unsigned char *)calloc(M_NUM_BITS, sizeof(unsigned char));
  String *h_string_array = (String *)malloc(INIT_WORDS * sizeof(String));
  String *h_test_array = (String *)malloc(INIT_WORDS * sizeof(String));
  int h_misses[1];
  h_misses[0] = 0;

  // initialize array to empty strings
  for (int i = 0; i < INIT_WORDS; i++)
  {
    strcpy(h_string_array[i].word, "");
  }

  for (int i = 0; i < INIT_WORDS; i++)
  {
    strcpy(h_test_array[i].word, "");
  }

  // device data
  unsigned char *d_bf_array;
  String *d_string_array;
  String *d_test_array;
  int *d_misses;

  // time measurement
  float map_time, test_time = 0;
  cudaEvent_t start_map, stop_map, start_test, stop_test;
  checkCudaErrors(cudaEventCreate(&start_map));
  checkCudaErrors(cudaEventCreate(&stop_map));
  checkCudaErrors(cudaEventCreate(&start_test));
  checkCudaErrors(cudaEventCreate(&stop_test));

  // open files
  FILE *map_fp = fopen(argv[1], "r");
  if (map_fp == NULL)
  {
    printf("Failed to open file1. \n");
    exit(1);
  }

  FILE *test_fp = fopen(argv[2], "r");
  if (test_fp == NULL)
  {
    printf("Failed to open file2. \n");
    exit(1);
  }

  // read in file1
  int num_words_mapped = fileToArray(map_fp, &h_string_array);
  int num_words_test = fileToArray(test_fp, &h_test_array);

  // allocate device arrays for map kernel
  checkCudaErrors(cudaMalloc((void **)&d_string_array, num_words_mapped * sizeof(String)));
  checkCudaErrors(cudaMemcpy(d_string_array, h_string_array, num_words_mapped * sizeof(String), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_test_array, num_words_test * sizeof(String)));
  checkCudaErrors(cudaMemcpy(d_test_array, h_test_array, num_words_test * sizeof(String), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_bf_array, M_NUM_BITS * sizeof(unsigned char)));
  checkCudaErrors(cudaMemcpy(d_bf_array, h_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyHostToDevice));

  // map words to Bloom filter
  checkCudaErrors(cudaEventRecord(start_map));
  cuda_mapToBloom<<<ceil(num_words_mapped / BLOCK_SIZE), BLOCK_SIZE>>>((unsigned char *)d_bf_array,
                                                                      (String *)d_string_array,
                                                                      num_words_mapped);
  checkCudaErrors(cudaEventRecord(stop_map));

  // get running time of map
  checkCudaErrors(cudaEventSynchronize(stop_map));
  checkCudaErrors(cudaEventElapsedTime(&map_time, start_map, stop_map));

  // allocate device data for test kernel
  checkCudaErrors(cudaMalloc((void **)&d_misses, sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_misses, h_misses, sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  // check if words are in Bloom filter
  checkCudaErrors(cudaEventRecord(start_test));
  cuda_countMisses<<<ceil(num_words_test / BLOCK_SIZE), BLOCK_SIZE>>>((unsigned char *)d_bf_array,
                                                                       (String *)d_test_array,
                                                                       (int *)d_misses, num_words_test);
  checkCudaErrors(cudaEventRecord(stop_test));

  // get running time of test
  checkCudaErrors(cudaEventSynchronize(stop_test));
  checkCudaErrors(cudaEventElapsedTime(&test_time, start_test, stop_test));

  // get resulting Bloom filter
  checkCudaErrors(cudaMemcpy(h_bf_array, d_bf_array, M_NUM_BITS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_misses, d_misses, sizeof(int), cudaMemcpyDeviceToHost));

  // print run time info
  printInfo(num_words_mapped, num_words_test, map_time, test_time, *h_misses);

  // cleanup
  cudaEventDestroy(start_map);
  cudaEventDestroy(stop_map);
  cudaEventDestroy(start_test);
  cudaEventDestroy(stop_test);
  cudaFree(d_bf_array);
  cudaFree(d_string_array);
  cudaFree(d_test_array);
  cudaFree(d_misses);
  free(h_bf_array);
  free(h_string_array);
  free(h_test_array);

  return 0;
}
