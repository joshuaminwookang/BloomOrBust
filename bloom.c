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
#include "bloom.h"

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
 * Reads words from array and adds them to Bloom filter.
 */
void addWordsFromArray(String *words, int num,  unsigned char *filter)
{
    for (int i = 0; i < num; i++)
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
    String *string_array = (String *)malloc(INIT_WORDS * sizeof(String));
    for (int i = 0; i < INIT_WORDS; i++)
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

    // measure time
    clock_t start, end;
    double cpu_time_used;

    printf("Made it here\n");

    // add words from file 1
    int num_words = fileToArray(add_fp, &string_array);

    printf("%d\n", num_words);

    start = clock();
    addWordsFromArray(string_array, num_words, bloom_filter_array);
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time to add to BF: %f ms\n", cpu_time_used);

    // check if words in file 2 are in Bloom filter
    printf("Misses: %d\n", countMissFromFile(check_fp, bloom_filter_array));

    free(bloom_filter_array);
    free(string_array);

    return 0;
}