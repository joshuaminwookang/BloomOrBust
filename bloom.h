/*
 * Bloom filter Header.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>

#define BUF_SIZE 100    // max size of word

typedef struct String {
    char word[BUF_SIZE];
} String;

/*
 * Reads words from file into array
 */
void fileToArray(FILE *fp, String* words) {
    char buffer[BUF_SIZE];

    int i = 0;
    while (fscanf(fp, "%s", buffer) == 1) {
        strcpy(words[i++].word, buffer);
    }
}