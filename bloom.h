/*
 * Bloom filter Header.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <ctype.h>

#define BUF_SIZE 100    // max size of word
#define M_NUM_BITS 1000 // number of elements in Bloom filter
#define K_NUM_HASH 5    // number of hash functions
#define HASH_NUM 5381   // number used for hash function
#define MAX_WORDS 500

typedef struct String
{
    char word[BUF_SIZE];
} String;

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
 * Removes punctuation characters
 */
void removePunct(char *str)
{
    char buffer[strlen(str) + 10];
    int j = 0;

    //make sure buffer is cleared
    memset(buffer, 0, sizeof(buffer));

    //only add non punctuation chars
    for (int i = 0; i < strlen(str); i++)
    {
        if (!ispunct(str[i]))
        {
            buffer[j++] = str[i];
        }
    }

    //copy temp buffer into string
    strcpy(str, buffer);
}

/*
 * Reads words from file into array
 */
void fileToArray(FILE *fp, String *words)
{
    char buffer[BUF_SIZE];

    int i = 0;
    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
        strcpy(words[i++].word, buffer);
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
