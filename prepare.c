/*
 * STN-Labz native language model — byte tokenizer and corpus preparation.
 * AI-assisted implementation; operator approval: Milestone 2.
 * This program does not train, serve, or modify an agent.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cap input to 16 MiB during this experimental stage. */
#define MAX_INPUT (16u * 1024u * 1024u)
#define SEED UINT64_C(20261020)
#define FNV_OFFSET UINT64_C(14695981039346656037)
#define FNV_PRIME UINT64_C(1099511628211)

/* Derive a repeatable 64-bit hash for provenance and line assignment. */
static uint64_t hash_bytes(const unsigned char *bytes, size_t length, uint64_t seed)
{
    uint64_t hash = FNV_OFFSET ^ seed;
    for (size_t i = 0; i < length; ++i) {
        hash = (hash ^ bytes[i]) * FNV_PRIME;
    }
    return hash;
}

/* Validate UTF-8 (including overlong, surrogate, and out-of-range sequences). */
static int valid_utf8(const unsigned char *s, size_t n)
{
    for (size_t i = 0; i < n;) {
        unsigned int cp;
        unsigned int min;
        unsigned char first = s[i++];
        int more;
        if (first < 0x80) {
            if (first == 0) return 0; /* no embedded NUL in text corpus */
            continue;
        }
        if (first >= 0xc2 && first <= 0xdf) {
            cp = first & 0x1f; more = 1; min = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            cp = first & 0x0f; more = 2; min = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            cp = first & 0x07; more = 3; min = 0x10000;
        } else return 0;
        if ((size_t) more > n - i) return 0;
        while (more--) {
            if ((s[i] & 0xc0) != 0x80) return 0;
            cp = (cp << 6) | (s[i++] & 0x3f);
        }
        if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return 0;
    }
    return 1;
}

/* Load the complete bounded source without modifying it. */
static unsigned char *load(const char *path, size_t *length)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t capacity = MAX_INPUT + 1u;
    unsigned char *data = malloc(capacity);
    if (!data) { fclose(f); return NULL; }
    size_t n = fread(data, 1, capacity, f);
    int error = ferror(f);
    int end = feof(f);
    fclose(f);
    if (error || !end || n == 0 || n > MAX_INPUT) {
        free(data); return NULL;
    }
    *length = n;
    return data;
}

/* Prepare byte tokens: each original UTF-8 byte is one token ID (0..255). */
int main(int argc, char **argv)
{
    if (argc != 5 || strcmp(argv[1], "prepare") != 0) {
        fprintf(stderr, "usage: %s prepare <authorized-utf8.txt> <train.bin> <validation.bin>\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[2], argv[3]) == 0 || strcmp(argv[2], argv[4]) == 0 ||
        strcmp(argv[3], argv[4]) == 0) {
        fputs("input and output paths must differ\n", stderr);
        return 2;
    }
    size_t n = 0;
    unsigned char *data = load(argv[2], &n);
    if (!data || !valid_utf8(data, n)) {
        fputs("source missing, empty, invalid UTF-8, or larger than 16 MiB\n", stderr);
        free(data);
        return 1;
    }
    /* Complete newline-delimited records only. Avoid splitting records across sets. */
    if (data[n-1] != '\n') {
        fputs("source must end with newline\n", stderr);
        free(data); return 1;
    }
    size_t lines = 0, train_n = 0, valid_n = 0;
    for (size_t i = 0, start = 0; i < n; ++i) {
        if (data[i] != '\n') continue;
        size_t length = i + 1u - start;
        if (length == 1u) { fputs("empty records not allowed\n", stderr); free(data); return 1; }
        ++lines;
        if (hash_bytes(data + start, length, SEED) % 5u == 0u) valid_n += length;
        else train_n += length;
        start = i + 1u;
    }
    if (lines < 5 || !train_n || !valid_n) {
        fputs("need at least five records and nonempty train/validation partitions\n", stderr);
        free(data); return 1;
    }
    /* Refuse to overwrite existing paths. Inputs are read-only. */
    FILE *train = fopen(argv[3], "rb");
    if (train) { fclose(train); fputs("train output exists\n", stderr); free(data); return 1; }
    FILE *valid = fopen(argv[4], "rb");
    if (valid) { fclose(valid); fputs("validation output exists\n", stderr); free(data); return 1; }
    train = fopen(argv[3], "wb");
    if (!train) { perror("train output"); free(data); return 1; }
    valid = fopen(argv[4], "wb");
    if (!valid) {
        perror("validation output"); fclose(train); remove(argv[3]); free(data); return 1;
    }
    int failed = 0;
    for (size_t i = 0, start = 0; i < n; ++i) {
        if (data[i] != '\n') continue;
        size_t length = i + 1u - start;
        FILE *target = hash_bytes(data + start, length, SEED) % 5u == 0u ? valid : train;
        if (fwrite(data + start, 1, length, target) != length) failed = 1;
        start = i + 1u;
    }
    if (fclose(train) != 0) failed = 1;
    if (fclose(valid) != 0) failed = 1;
    if (failed) {
        fputs("output write failed\n", stderr);
        remove(argv[3]); remove(argv[4]); free(data); return 1;
    }
    printf("format=raw_utf8_bytes vocabulary=256 seed=%llu input_fnv64=%016llx records=%zu train_bytes=%zu validation_bytes=%zu\n",
           (unsigned long long)SEED,
           (unsigned long long)hash_bytes(data, n, 0),
           lines, train_n, valid_n);
    free(data);
    return 0;
}
