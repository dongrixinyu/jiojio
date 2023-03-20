#ifndef _WCHAR_T_HASH_DICT_H
#define _WCHAR_T_HASH_DICT_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <wchar.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

typedef struct DictHashNode_Struct DictHashNode;

struct DictHashNode_Struct
{
    wchar_t *sKey;
    int nValue;
    DictHashNode *pNext;
};

DictHashNode **dict_hash_table_init(int hashTableMaxSize);
unsigned int dict_hash_table_hash_str(const wchar_t *skey);
void dict_hash_table_insert(
    DictHashNode **dictHashTable, const wchar_t *skey, int nvalue, int hashTableMaxSize);
void dict_hash_table_remove(DictHashNode **dictHashTable, const wchar_t *skey, int hashTableMaxSize);
int dict_hash_table_lookup(DictHashNode **dictHashTable, const wchar_t *skey, int hashTableMaxSize);
void dict_hash_table_release(DictHashNode **dictHashTable, int hashTableMaxSize);
void dict_distribution_statistics(DictHashNode **dictHashTable, int hashTableMaxSize, int hashTableSize);

#endif
