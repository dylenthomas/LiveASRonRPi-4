#ifndef KEYWORDCONFPARSER_H
#define KEYWORDCONFPARSER_H

#include <stdio.h> 
#include <assert.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>

struct item {
    uint64_t value;
    struct item* next;
};

struct keywordHM {
    struct item** items;
    int len;
};

uint64_t djb2Hash(char* str);
static void fileOpErrorCheck(int opperationReturn, FILE* fptr);
static int peekNext(FILE* fptr);
static int getKey(FILE* fptr, char key[], int longest_key);
struct keywordHM createKeywordHM(const char* path);

#endif
