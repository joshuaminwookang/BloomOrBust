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

    rewind(fp);

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
        mapToBloom(filter, words[i].word);
    }
}

int countMissFromArray(String *words, int num,  unsigned char *filter) {
    
    int count = 0;

    for (int i = 0; i < num; i++)
    {
        if (!checkBloom(filter, words[i].word)) {
            count++;
        }
    }

    return count;
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

    // initialize arrays of Strings
    String *add_array = (String *)malloc(INIT_WORDS * sizeof(String));
    String *check_array = (String *)malloc(INIT_WORDS * sizeof(String));
    

    // open files
    FILE *add_fp = fopen(argv[1], "r");
    if (add_fp == NULL)
    {
        printf("Failed to open %s. \n", argv[1]);
        exit(1);
    }

    FILE *check_fp = fopen(argv[2], "r");
    if (check_fp == NULL)
    {
        printf("Failed to open %s. \n", argv[2]);
        exit(1);
    }

    // measure time 
    clock_t start, end;
    double add_time, check_time;

    // add words from files
    int num_words_added = fileToArray(add_fp, &add_array);
    int num_words_check = fileToArray(check_fp, &check_array);

    int misses;

    // add words to Bloom filter
    start = clock();
    addWordsFromArray(add_array, num_words_added, bloom_filter_array);
    end = clock();
    add_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // check if words in file 2 are in Bloom filter
    start = clock();
    misses = countMissFromArray(check_array, num_words_check, bloom_filter_array);
    end = clock();
    check_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // print out info
    printInfo(num_words_added, num_words_check, add_time, check_time, misses);

    free(bloom_filter_array);
    free(add_array);
    free(check_array);

    return 0;
}
