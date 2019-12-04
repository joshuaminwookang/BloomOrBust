/*
 * Bloom filter Header. 
 * 
 * Contains functions to be used by both parallel
 * and sequential implementations of the Bloom filter.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <ctype.h>

#define BUF_SIZE 100     // max size of word
#define M_NUM_BITS 20000 // number of elements in Bloom filter
#define K_NUM_HASH 5    // number of hash functions
#define HASH_NUM 5381   // number used for hash function
#define INIT_WORDS 512
#define MAX_WORDS 

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
 * Reads words from file into array.
 * 
 * Returns the number of words read in.
 */
int fileToArray(FILE *fp, String **words)
{
    char buffer[BUF_SIZE];
    int size = INIT_WORDS;

    rewind(fp);

    int i = 0;
    while (fscanf(fp, "%s", buffer) != EOF)
    {
        removePunct(buffer);
        strcpy((*words)[i++].word, buffer);

        if (i >= size) {
            size = size * 2;

            *words = (String *)realloc(*words, size * sizeof(String));

        }
    }

    return i;
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
            free(hashes);
            return 0;
        }
    }

    free(hashes);

    return 1;
}

/*
 * Checks if words from file are in Bloom filter.
 */
void checkWordsFromFile(FILE *fp, unsigned char *filter)
{
    char buffer[BUF_SIZE];

    rewind(fp);

    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
        printf("%d: %s\n", checkBloom(filter, buffer), buffer);
    }
}

/*
 * Counts the number of words missing in a file.
 * 
 * Returns that count.
 */
int countMissFromFile(FILE *fp, unsigned char *filter)
{
    char buffer[BUF_SIZE];
    int count = 0;

    rewind(fp);

    while (fscanf(fp, "%s", buffer) == 1)
    {
        removePunct(buffer);
    
        if (!checkBloom(filter, buffer)) {
            count++;
        }
    }

    return count;
}

void printFilter(unsigned char *filter) {
    for (int i = 0; i < M_NUM_BITS; i++) {
        printf("%d", filter[i]);
    }
    printf("\n");
}
