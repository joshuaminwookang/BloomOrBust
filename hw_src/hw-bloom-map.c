
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
#include "small_data.h"
#include "medium_data.h"
#include "big_data.h"

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
 * Initializes / resets Bloom filter hardware accelerator 
 */
static inline unsigned long hw_initBloom()
{
    unsigned long rd;
    // asm volatile ("fence");
	ROCC_INSTRUCTION(2, 0);
    // asm volatile ("fence");
	return rd ;
}

/*
 * Maps (already hashed) word to Bloom filter
 * @ params: hash value of input string to be mapped
 * @ returns: hash value of input string
 */
static inline unsigned long hw_mapToBloom(long hash)
{
    unsigned long rd;
    // asm volatile ("fence");
	ROCC_INSTRUCTION_DS(2, rd, hash, 1);
    // asm volatile ("fence");
	return rd;
}

/*
 * Tests if word is in Bloom filter
 * @ params: hash value of string to be tested against BF
 * @ returns: current miss count
 */
static inline unsigned long hw_testBloom(long hash)
{
    unsigned long rd;
    // asm volatile ("fence");
	ROCC_INSTRUCTION_DS(2, rd, hash, 2);
    // asm volatile ("fence");
	return rd;
}

/*
 * Using HW accelerator:
 * reads words from array and map them to Bloom filter.
 */
void hw_mapWordsFromArray(int num)
{
    for (int i = 0; i < num; i++)
    {
       unsigned long returnValue ; 
    //    
        #ifdef TINY       
        printf("Word to MAP: %s with hash value :%lu\n",tiny0[i], hashstring(tiny0[i]));
        returnValue = hw_mapToBloom(hashstring(tiny0[i]));
        printf("HW Map Function returned: %lu\n", returnValue);
        #endif
        #ifdef SMALL       
        returnValue = hw_mapToBloom(hashstring(small[i]));
        #endif
        #ifdef MEDIUM       
        returnValue = hw_mapToBloom(hashstring(medium[i]));
        #endif
        #ifdef BIG       
        returnValue = hw_mapToBloom(hashstring(large[i]));
        #endif
    }
}

/* (Using HW accelerator)
 * Counts number of misses from tests
 */
int hw_countMissFromArray(int num)
{

    int count = 0;

    for (int i = 0; i < num; i++)
    {
        #ifdef TINY
        printf("Word to TEST: %s with hash value : %lu\n",tiny1[i], hashstring(tiny1[i]));
        count = hw_testBloom(hashstring(tiny1[i]));
        printf("Current miss count: %d\n", count);
        #else
        count = hw_testBloom(hashstring(big[i]));
        #endif 
    }

    return count;
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


/*
 * Test script 
 */
int main(void)
{
    unsigned long start, end;
    int hw_misses = 0;
    int sw_misses = 0;


    // Initalize BF Accelerator
    asm volatile ("fence");
    hw_initBloom();

    // Initialize SW bloom filter array
    // memset(bloom_filter_array, 0, M_NUM_BITS);
    for (int i = 0; i < M_NUM_BITS; i++)
    {
        bloom_filter_array[0] = 0;
    }

    // HW: MAP
    start = rdcycle();                                                                                                                                      
    asm volatile ("fence");
    #ifdef TINY
        hw_mapWordsFromArray(TINY);
    #endif 
    #ifdef SMALL
        hw_mapWordsFromArray(SMALL);
    #endif 
    #ifdef MEDIUM
        hw_mapWordsFromArray(MEDIUM);
    #endif 
    #ifdef BIG
        hw_mapWordsFromArray(BIG);
    #endif 
    asm volatile ("fence");
    end = rdcycle();
    printf("MAP execution took %lu cycles\n", end - start);

    // HW: TEST
    start = rdcycle();  
    asm volatile ("fence");
    #ifdef TINY
        hw_misses = hw_countMissFromArray(TINY);
    #else
        hw_misses = hw_countMissFromArray(TEST_SIZE);
    #endif 
    asm volatile ("fence");
    end = rdcycle();   
    printf("TEST execution took %lu cycles\n", end - start);
    // print out test results
    printf(" HW Miss: %d: \n", hw_misses);

    // SW: Map
    start = rdcycle(); 
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
    end = rdcycle();  
    printf("SW MAP execution took %lu cycles\n", end - start); 

    // SW: TEST
    start = rdcycle(); 
    // test if words in file 2 are in Bloom filter
    sw_misses = countMissFromArray(TEST_SIZE);
    end = rdcycle(); 

    // print out info
    printf("SW TEST execution took %lu cycles\n", end - start); 
    printf("Software Misses: %d\n", sw_misses);

    return 0;
}
