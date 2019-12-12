
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>
#include "rocc.h"
#include "encoding.h"
#include "compiler.h"

#ifdef __linux
#include <sys/mman.h>
#endif

#define BUF_SIZE 100     // max size of word
#define M_NUM_BITS 20000 // number of elements in Bloom filter
#define K_NUM_HASH 5     // number of hash functions
#define HASH_NUM 5381    // number used for hash function
#include "small_data.h"

// #define TINY 11
#define TINYV2 30
// #define TINYV3 50

/* global Bloom bit array */
unsigned char bloom_filter_array[M_NUM_BITS];


/*
 * Hash function for a string using Horner's Rule.
 * Given a string, returns a number.
 */
unsigned long hashstring(char* word)
{
    unsigned char *str = (unsigned char *)word;
    unsigned long hash = HASH_NUM;

    // while there are still chars in the word
    while (*str)
    {
        // hash = (hash * 32) + hash + current char in word
        hash = ((hash << 5) + hash) + *(str++);
    }

    return hash;
}

/*
 * map word to bloom filter.
 * Places 1 in filter at indices that given word maps to.
 */
void mapToBloom(int index)
{
    long x = hashstring(medium[index]); 
    
    long y = x >> 4;

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        x = (x + y) % M_NUM_BITS; // ith hash value
        y = (y + i) % M_NUM_BITS; // displacement
        bloom_filter_array[x] = 1;
    }
}

/*
 * Reads words from array and maps them to Bloom filter.
 */
void mapWordsFromArray(int num)
{
    for (int i = 0; i < num; i++)
    {
        mapToBloom(i);
    }
}

/*
 * tests if word is in bloom filter.
 * Tests if there is a 1 in filter at indices 
 * that given word maps to.
 *
 * Returns 1 if search is positive, 0 if negative.
 */
int testBloom(int index)
{
    #ifdef TINY
    long x = hashstring(tiny1[index]); 
    #endif
    #ifdef SMALL
    long x = hashstring(small[index]); 
    #endif
    #ifdef MEDIUM
    long x = hashstring(medium[index]); 
    #endif
    #ifdef BIG
    long x = hashstring(large[index]); 
    #endif
    long y = x >> 4; 

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        x = (x + y) % M_NUM_BITS; // ith hash value
        y = (y + i) % M_NUM_BITS; // displacement

        if (!bloom_filter_array[x])
        {
            return 0;
        }
    }

    return 1;
}

int countMissFromArray(int num)
{
    int count = 0;

    for (int i = 0; i < num; i++)
    {
        if (!testBloom(i))
        {
            count++;
        }
    }

    return count;
}


/*
 * Test script 
 */
int main(void)
{
    unsigned long start, end;
    int sw_misses = 0;

    // Initialize SW bloom filter array
    memset(bloom_filter_array, 0, M_NUM_BITS);
    // for (int i = 0; i < M_NUM_BITS; i++)
    // {
    //     bloom_filter_array[0] = 0;
    // }

    // SW: Map
    start = rdcycle(); 
    // map words to Bloom filter
    mapWordsFromArray(MAP_SIZE);
    end = rdcycle();  
    printf("SW MAP execution took %lu cycles\n", end - start); 

    // SW: TEST
    start = rdcycle(); 
    // test if words in file 2 are in Bloom filter
    #ifdef TINY
        sw_misses = countMissFromArray(TINY);
    #endif 
    #ifdef SMALL
        sw_misses = countMissFromArray(SMALL);
    #endif 
    #ifdef MEDIUM
        sw_misses = countMissFromArray(MEDIUM);
    #endif 
    #ifdef BIG
        sw_misses = countMissFromArray(BIG);
    #endif 
    end = rdcycle(); 

    // print out info
    printf("SW TEST execution took %lu cycles\n", end - start); 
    printf("Software Misses: %d\n", sw_misses);

    return 0;
}
