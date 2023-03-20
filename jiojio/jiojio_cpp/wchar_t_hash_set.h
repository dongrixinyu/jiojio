#ifndef _WCHAR_T_HASH_SET_H
#define _WCHAR_T_HASH_SET_H

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

typedef struct SetHashNode_Struct SetHashNode;

struct SetHashNode_Struct
{
    wchar_t *sKey;
    SetHashNode *pNext;
};

SetHashNode **set_hash_table_init(int hashTableMaxSize);

unsigned int set_hash_table_hash_str(const wchar_t *skey);
void set_hash_table_insert(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize);
void set_hash_table_remove(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize);
int set_hash_table_lookup(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize);
void set_hash_table_release(SetHashNode **setHashTable, int hashTableMaxSize);

// statistics
double kl_divergence(int count_table[], int hashTableMaxSize, int hashTableSize);
float detect_empty_bucket(int count_table[], int hashTableMaxSize);
void distribution_statistics(SetHashNode **setHashTable, int hashTableMaxSize, int hashTableSize);

#endif
