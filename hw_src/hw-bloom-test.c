
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
#define DEFAULT 10000
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
        returnValue = hw_mapToBloom(hashstring(small[i]));
    }
    return;
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
        #endif 
        #ifdef SMALL
        count = hw_testBloom(hashstring(small[i]));
        #endif
        #ifdef MEDIUM
        count = hw_testBloom(hashstring(medium[i]));
        #endif
        #ifdef BIG
        count = hw_testBloom(hashstring(big[i]));
        #endif

    }
    return count;
}

/* (Using HW accelerator)
 * Counts number of misses from tests
 */
int hw_dummy_countMissFromArray(int num)
{

    int count = 0;

    for (int i = 0; i < num; i++)
    {
        printf("Word to TEST: %s with hash value :%lu\n",tiny0[i], hashstring(tiny0[i]));
        count = hw_testBloom(hashstring(tiny0[i]));
        printf("Current miss count: %d\n", count);
    }

    return count;
}

/*
 * Test script 
 */
int main(void)
{
    unsigned long start, end;
    int hw_misses;

    // Initalize BF Accelerator
    asm volatile ("fence");
    hw_initBloom();

    // Compute Map with accelerator   
    start = rdcycle();                                                                                                                                      
    asm volatile ("fence");
    // HW: map words to Bloom filter
    hw_mapWordsFromArray(10000);
    asm volatile ("fence");
    end = rdcycle();
    printf("MAP execution took %lu cycles\n", end - start);

    // // HW: test if words in file 2 are in Bloom filter
    // start = rdcycle();  
    // asm volatile ("fence");
    // hw_misses = hw_dummy_countMissFromArray(NUM_WORDS);
    // asm volatile ("fence");
    // end = rdcycle();    
    // printf("DUMMY execution took %lu cycles\n", end - start);
    // // print out test results
    // printf(" HW Miss: %d: \n", hw_misses);

    // HW: test if words in file 2 are in Bloom filter
    start = rdcycle();  
    asm volatile ("fence");
    #ifdef TINY
        hw_misses = hw_countMissFromArray(TINY);
    #endif 
    #ifdef SMALL
        hw_misses = hw_countMissFromArray(SMALL);
    #endif 
    #ifdef MEDIUM
        hw_misses = hw_countMissFromArray(MEDIUM);
    #endif 
    #ifdef BIG
        hw_misses = hw_countMissFromArray(BIG);
    #endif 
    asm volatile ("fence");
    end = rdcycle();   

    printf("TEST execution took %lu cycles\n", end - start);
    // print out test results
    printf(" HW Miss: %d: \n", hw_misses);

    return 0;
}
