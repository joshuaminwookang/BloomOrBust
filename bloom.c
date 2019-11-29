/*
 * Bloom filter with sequential hash functions.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bloom.h"

/*
 * Hash function for a string using Horner's Rule.
 * Given a string, returns a number.
 */
unsigned long hashstring(char *word)
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
 * Hash function: Finds indicies that a word maps to.
 * @params long[] hashes is the array of indices 
 * @params char* word is the word we want to add
 */
void hash(long *hashes, char *word)
{
    unsigned long x = hashstring(word);
    unsigned long y = x >> 4;

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        x = (x + y) % M_NUM_BITS;
        y = (y + i) % M_NUM_BITS;
        hashes[i] = x;
    }
}

/*
 * Add word to bloom filter.
 * Places 1 in filter at indices that given word maps to.
 */
void mapToBloom(unsigned char *filter, char *word)
{
    long *hashes = (long *)calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        filter[hashes[i]] = 1;
    }
}

/*
 * Checks if word is in bloom filter.
 * Tests if there is a 1 in filter at indices 
 * that given word maps to.
 *
 * Returns 1 if search is positive, 0 if negative.
 */
int checkBloom(unsigned char *filter, char *word)
{
    long *hashes = (long *)calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++)
    {
        if (!filter[hashes[i]])
        {
            return 0;
        }
    }

    return 1;
}

/*
 * Reads words from file and adds them to Bloom filter.
 */
void addWordsFromFile(FILE *fp, unsigned char *filter)
{
    char buffer[BUF_SIZE];

    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
        mapToBloom(filter, buffer);
    }
}

/*
 * Checks if words from file are in Bloom filter.
 */
void checkWordsFromFile(FILE *fp, unsigned char *filter)
{
    char buffer[BUF_SIZE];

    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
        printf("%d: %s\n", checkBloom(filter, buffer), buffer);
    }
}

/*
 * Reads words from array and adds them to Bloom filter.
 */
void addWordsFromArray(String *words, unsigned char *filter)
{
    for (int i = 0; i < MAX_WORDS; i++)
    {

        // check if there are no more words
        if (!strcmp(words[i].word, ""))
            return;

        mapToBloom(filter, words[i].word);
    }
}

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        printf("Usage: ./bloom WordsToAdd WordsToCheck\n");
        exit(1);
    }

    // initialize bloom filter array
    unsigned char *bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));

    // initialize array of Strings
    String *string_array = (String *)malloc(MAX_WORDS * sizeof(String));
    for (int i = 0; i < MAX_WORDS; i++)
    {
        strcpy(string_array[i].word, "");
    }

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

    // add words from file 1
    fileToArray(add_fp, string_array);
    addWordsFromArray(string_array, bloom_filter_array);

    // check if words in file 2 are in Bloom filter
    checkWordsFromFile(check_fp, bloom_filter_array);

    free(bloom_filter_array);
    free(string_array);

    return 0;
}