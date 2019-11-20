#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

 #define M_NUM_BITS 100
 #define K_NUM_HASH 5

//  typedef struct bloom_filter
//  {
//      /* K hash functions are used, seeded by caller's seed */
//      int         k_hash_funcs;
//      unsigned long      seed;
//      /* m is bitset size, in bits.  Must be a power of two <= 2^32.  */
//      unsigned long      m_bits;
//      unsigned char bitset[ARRAY_LENGTH ];
//  }bloom_filter;

//  // method to initialize Bloom filter
// bloom_filter * 
// bloom_create(int m, int k, unsigned long seed) {
//     bloom_filter *myBloomFilter;
//     myBloomFilter = (bloom_filter*) calloc(1, sizeof(bloom_filter));

//     myBloomFilter-> k_hash_funcs = k;
//     myBloomFilter->seed = seed;
//     myBloomFilter->m_bits = m;
// }

// readfile function

unsigned long hashstring(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

// hash function
// @params long[] hashes is the array of indices 
void hash(long *hashes, char* word) {
    unsigned long x = hashstring(word);
    unsigned long y = hashstring(word) >> 4;

    for(int i=0; i<K_NUM_HASH; i++) {
        x = (x+y) % M_NUM_BITS;
        y = (y+i) % M_NUM_BITS;
        hashes[i] = x;
    }
}

// mapBloom
void mapBloom (unsigned char *filter, char* word) {
    long *hashes = (long *) calloc(K_NUM_HASH, sizeof(long));
    hash(hashes, word);

    for (int i = 0; i < K_NUM_HASH; i++) {
        filter[hashes[i]] = 1;
    }
}


// checkBloom



int main(int argc, char** argv) {
    srand(time(NULL));
    unsigned char *bloom_filter_array = calloc(M_NUM_BITS, sizeof(unsigned char));
    char *test = "Andrew Thai";
    mapBloom(bloom_filter_array, test);

    for (int i = 0; i< M_NUM_BITS; i++){
        if (bloom_filter_array[i] == 1) {printf("INdex! %d \n", i);}
    }

    return 0;
}