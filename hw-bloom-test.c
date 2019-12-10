
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
#define NUM_WORDS 20

// hard-coded test inputs
static char tiny0 [20][BUF_SIZE] = {
    "words", "in", "this," 
    "file",  "will", "be", "added", "to,","the", "bloom" , "filter"
};

static char tiny1 [20][BUF_SIZE] = {
    "these", "words", "may", "or," 
    "may",  "not", "be", "in","the", "bloom" , "filter"
};

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
	ROCC_INSTRUCTION(2, 0);
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
	ROCC_INSTRUCTION_DS(2, rd, hash, 1);
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
	ROCC_INSTRUCTION_DS(2, rd, hash, 2);
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
       printf("Word to MAP: %s with hash value :%lu\n",tiny0[i], hashstring(tiny0[i]);
       returnValue = hw_mapToBloom(hashstring(tiny0[i]));
       printf("HW Map Function returned: %lu\n", returnValue);
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
        printf("Word to TEST: %s with hash value :%lu\n",tiny1[i], hashstring(tiny1[i]);
        count = hw_testBloom(hashstring(tiny1[i]));
        printf("Current miss count: %d\n", count);
    }

    return count;
}



/*
 * Test script 
 */
int main(void)
{
    int hw_misses;
    // HW: map words to Bloom filter
    hw_mapWordsFromArray(NUM_WORDS);

    // HW: test if words in file 2 are in Bloom filter
    hw_misses = hw_countMissFromArray(NUM_WORDS);

    // print out test results
    printf(" HW Miss: %d: \n", hw_misses);
    return 0;
}
