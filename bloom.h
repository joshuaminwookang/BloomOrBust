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
