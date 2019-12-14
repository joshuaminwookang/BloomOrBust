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
 * map word to bloom filter.
 * Places 1 in filter at indices that given word maps to.
 */
void mapToBloom(unsigned char *filter, char *word)
{
    long *hashes = (long *)calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    // set the bits at the hash value indices to 1
    for (int i = 0; i < K_NUM_HASH; i++)
    {
        filter[hashes[i]] = 1;
    }
}

/*
 * Reads words from file and maps them to Bloom filter.
 */
void mapWordsFromFile(FILE *fp, unsigned char *filter)
{
    char buffer[BUF_SIZE];

    rewind(fp); // make sure we start at beginning

    // read in each word, remove punctuation, and map it
    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
        mapToBloom(filter, buffer);
    }
}

/*
 * Reads words from array and maps them to Bloom filter.
 */
void mapWordsFromArray(String *words, int num, unsigned char *filter)
{
    for (int i = 0; i < num; i++)
    {
        mapToBloom(filter, words[i].word);
    }
}

/*
 * Counts the number of misses by testing each word in array.
 */
int countMissFromArray(String *words, int num, unsigned char *filter)
{
    int count = 0;

    for (int i = 0; i < num; i++)
    {
        if (!testBloom(filter, words[i].word))
        {
	  count++; // miss
        }
    }

    return count;
}

/*
 * Reads files and maps words from file 1 to Bloom filter.
 * Tests words in file 2.
 * 
 * Reports running time info at the end.
 */
int main(int argc, char **argv)
{

    if (argc != 3)
    {
        printf("Usage: ./bloom WordsToMap WordsToTest\n");
        exit(1);
    }

    // initialize bloom filter array
    unsigned char *bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));

    // initialize arrays of Strings
    String *map_array = (String *)malloc(INIT_WORDS * sizeof(String));
    String *test_array = (String *)malloc(INIT_WORDS * sizeof(String));

    // open files
    FILE *map_fp = fopen(argv[1], "r");
    if (map_fp == NULL)
    {
        printf("Failed to open %s. \n", argv[1]);
        exit(1);
    }

    FILE *test_fp = fopen(argv[2], "r");
    if (test_fp == NULL)
    {
        printf("Failed to open %s. \n", argv[2]);
        exit(1);
    }

    // measure time
    clock_t start, end;
    double map_time, test_time;

    // map words from files
    int num_words_mapped = fileToArray(map_fp, &map_array);
    int num_words_test = fileToArray(test_fp, &test_array);

    int misses;

    // map words to Bloom filter
    start = clock();
    mapWordsFromArray(map_array, num_words_mapped, bloom_filter_array);
    end = clock();
    map_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // test if words in file 2 are in Bloom filter
    start = clock();
    misses = countMissFromArray(test_array, num_words_test, bloom_filter_array);
    end = clock();
    test_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    // print out info
    printInfo(num_words_mapped, num_words_test, map_time, test_time, misses);

    free(bloom_filter_array);
    free(map_array);
    free(test_array);

    return 0;
}
