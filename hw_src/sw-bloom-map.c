
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
#define TINY 11
// #define SMALL 10000
// #define MEDIUM 466551
// #define BIG 1095695
#define TEST_SIZE 1095695

#ifdef TINY
#include "small_data.h"
#endif
#ifdef SMALL
#include "small_data.h"
#endif
#ifdef MEDIUM
#include "medium_data.h"
#endif
#include "big_data.h"


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
void mapToBloom(unsigned char * filter,int index)
{
    #ifdef TINY
    long x = hashstring(tiny0[index]); 
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
        filter[x] = 1;
    }
}

/*
 * Reads words from array and maps them to Bloom filter.
 */
void mapWordsFromArray(unsigned char * filter, int num)
{
    for (int i = 0; i < num; i++)
    {
        mapToBloom(filter, i);
    }
}

/*
 * tests if word is in bloom filter.
 * Tests if there is a 1 in filter at indices 
 * that given word maps to.
 *
 * Returns 1 if search is positive, 0 if negative.
 */
int testBloom(unsigned char * filter,int index)
{
    long x = hashstring(big[index]); 
    long y = x >> 4; 

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        x = (x + y) % M_NUM_BITS; // ith hash value
        y = (y + i) % M_NUM_BITS; // displacement

        if (!filter[x])
        {
            return 0;
        }
    }

    return 1;
}

int countMissFromArray(unsigned char * filter, int num)
{
    int count = 0;

    for (int i = 0; i < num; i++)
    {
        if (!testBloom(filter, i))
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

    /* SW Bloom bit array */
    unsigned char bloom_filter_array[M_NUM_BITS];   
    // Initialize SW bloom filter array
    memset(bloom_filter_array, 0, M_NUM_BITS);
    // for (int i = 0; i < M_NUM_BITS; i++)
    // {
    //     bloom_filter_array[0] = 0;
    // }

    // SW: Map
    start = rdcycle(); 
    // map words to Bloom filter
    #ifdef TINY
    mapWordsFromArray(&bloom_filter_array, TINY);
    #endif

    #ifdef SMALL
    mapWordsFromArray(&bloom_filter_array, SMALL);
    #endif

    #ifdef MEDIUM 
    mapWordsFromArray(&bloom_filter_array, MEDIUM);
    #endif

    #ifdef BIG 
    mapWordsFromArray(&bloom_filter_array, BIG);
    #endif
    end = rdcycle();  
    printf("SW MAP execution took %lu cycles\n", end - start); 

    // SW: TEST
    start = rdcycle(); 
    // test if words in file 2 are in Bloom filter
    sw_misses = countMissFromArray(&bloom_filter_array,TEST_SIZE);
    end = rdcycle(); 

    // print out info
    printf("SW TEST execution took %lu cycles\n", end - start); 
    printf("Software Misses: %d\n", sw_misses);

    return 0;
}
