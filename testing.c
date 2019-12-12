/*
 * Bloom filter with sequential hash functions.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "hw_src/small_data.h"
#include "hw_src/medium_data.h"
#include "hw_src/big_data.h"

#define BUF_SIZE 100     // max size of word
#define M_NUM_BITS 20000 // number of elements in Bloom filter
#define K_NUM_HASH 5     // number of hash functions
#define HASH_NUM 5381    // number used for hash function
 #define TINY 11
// #define SMALL 10000
// #define MEDIUM 466551
// #define BIG 1095695
#define TEST_SIZE 1095695

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
    // #ifdef TINY
    // long x = hashstring(tiny1[index]); 
    // #endif
    // #ifdef SMALL
    // long x = hashstring(small[index]); 
    // #endif
    // #ifdef MEDIUM
    // long x = hashstring(medium[index]); 
    // #endif
    // #ifdef BIG
    // long x = hashstring(large[index]); 
    // #endif
    long x = hashstring(big[index]); 
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

int main(int argc, char **argv)
{
    // initialize bloom filter array
    // memset(bloom_filter_array, 0, M_NUM_BITS);
    for (int i = 0; i < M_NUM_BITS; i++)
    {
        bloom_filter_array[0] = 0;
    }

    int misses;

    // map words to Bloom filter
    #ifdef TINY
    mapWordsFromArray(TINY);
    #endif

    #ifdef SMALL
    mapWordsFromArray(SMALL);
    #endif

    #ifdef MEDIUM 
    mapWordsFromArray(MEDIUM);
    #endif

    #ifdef BIG 
    mapWordsFromArray(BIG);
    #endif

    // test if words in file 2 are in Bloom filter
    misses = countMissFromArray(TEST_SIZE);

    // print out info
    printf("Software Misses: %d\n", misses);

    return 0;
}
