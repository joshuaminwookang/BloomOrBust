
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>
#include "rocc.h"
#include "bloom.h"
#include "encoding.h"
#include "compiler.h"

#ifdef __linux
#include <sys/mman.h>
#endif

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
 * Software version of function to map a word to bloom filter.
 * Places 1 in filter at indices that given word maps to.
 */
void sw_mapToBloom(unsigned char *filter, char *word)
{
    long *hashes = (long *)calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        filter[hashes[i]] = 1;
    }
}

/*
 * Using HW accelerator:
 * reads words from array and map them to Bloom filter.
 */
void hw_mapWordsFromArray(String *words, int num)
{
    for (int i = 0; i < num; i++)
    {
       hw_mapToBloom((int) hashstring(words[i].word));
    }
}

/* (Using C SW)
 * Reads words from array and maps them to Bloom filter.
 */
void sw_mapWordsFromArray(String *words, int num, unsigned char *filter)
{
    for (int i = 0; i < num; i++)
    {
        sw_mapToBloom(filter, words[i].word);
    }
}

/* (Using HW accelerator)
 * Counts number of misses from tests
 */
int hw_countMissFromArray(String *words, int num)
{

    int count = 0;

    for (int i = 0; i < num; i++)
    {
        count = hw_testBloom(hashstring(words[i].word));
    }

    return count;
}

/* (Using C SW)
 * Counts number of misses from tests
 */
int sw_countMissFromArray(String *words, int num, unsigned char *filter)
{

    int count = 0;

    for (int i = 0; i < num; i++)
    {
        if (!testBloom(filter, words[i].word))
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
    // initialize bloom filter array
    unsigned char *sw_bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));

    // initialize arrays of Strings
    String *words_to_map= (String *)malloc(INIT_WORDS * sizeof(String));
    String *words_to_test = (String *)malloc(INIT_WORDS * sizeof(String));

    // open files
    FILE *map_fp = fopen("tiny0", "r");
    if (map_fp == NULL)
    {
        printf("Failed to open %s. \n", "tiny0");
        exit(1);
    }

    FILE *test_fp = fopen("tiny1", "r");
    if (test_fp == NULL)
    {
        printf("Failed to open %s. \n", "tiny1");
        exit(1);
    }

    // map words from files
    int num_words_mapped = fileToArray(map_fp, &words_to_map);
    int num_words_test = fileToArray(test_fp, &words_to_test);

    int sw_misses;
    int hw_misses;


    // SW: map words to Bloom filter
    sw_mapWordsFromArray(words_to_map, num_words_mapped, sw_bloom_filter_array);

    // SW: test if words in file 2 are in Bloom filter
    sw_misses = sw_countMissFromArray(words_to_test, num_words_test, sw_bloom_filter_array);

    // HW: map words to Bloom filter
    hw_mapWordsFromArray(words_to_map, num_words_mapped);

    // HW: test if words in file 2 are in Bloom filter
    hw_misses = hw_countMissFromArray(words_to_test, num_words_test);

    
    // print out test results
    if (sw_misses == hw_misses){  
        printf(" TEST Successful \n HW Miss: %d SW MISS: \n", hw_misses, sw_misses);
    } else {
        printf(" TEST Failed \n HW Miss: %d SW MISS: \n", hw_misses, sw_misses);
        exit(1);
    }
    free(sw_bloom_filter_array);
    free(words_to_map);
    free(words_to_test);

    return 0;
}
