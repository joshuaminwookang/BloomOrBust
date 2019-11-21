/*
 * Bloom filter.
 *
 * (c) 2019 Josh Kang and Andrew Thai
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define M_NUM_BITS 100
#define K_NUM_HASH 5

#define BUF_SIZE 100



/*
 * Hash function for a string.
 * Given a string, returns a number.
 */
unsigned long hashstring(char *word) {
    unsigned char *str = (unsigned char*)word;
    unsigned long hash = 5381;
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    
    return hash;
}

/* 
 * Hash function: Finds indicies that a word maps to.
 * @params long[] hashes is the array of indices 
 * @params char* word is the word we want to add
 */
void hash(long *hashes, char* word) {
    unsigned long x = hashstring(word);
    unsigned long y = hashstring(word) >> 4;

    for(int i=0; i<K_NUM_HASH; i++) {
        x = (x+y) % M_NUM_BITS;
        y = (y+i) % M_NUM_BITS;
        hashes[i] = x;
    }
}

/*
 * Add word to bloom filter.
 * Places 1 in filter at indices that given word maps to.
 */
void mapToBloom (unsigned char *filter, char* word) {
    long *hashes = (long *) calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++) {
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
int checkBloom (unsigned char *filter, char* word) {
    long *hashes = (long *) calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++) {
         if (!filter[hashes[i]]) {
             return 0;
         }
    }

    return 1;
}

/*
 * Reads words from file and adds them to Bloom filter.
 */
void addWordsFromFile(FILE *fp, unsigned char* filter) {
    char buffer[BUF_SIZE];

    while (fscanf(fp, "%s", buffer) == 1) {
        printf("Read: %s\n", buffer);
        mapToBloom(filter, buffer);
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    unsigned char *bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));
    
    FILE *fp = fopen(argv[1], "r");    
    
    addWordsFromFile(fp, bloom_filter_array);

    char *test1 = "be";

    //mapToBloom(bloom_filter_array, test1);

    printf("Found word: %d \n", checkBloom(bloom_filter_array, test1));

    return 0;
}