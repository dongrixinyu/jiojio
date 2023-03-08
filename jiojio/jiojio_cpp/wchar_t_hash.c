/*
 * Author: puresky
 * Date: 2011/01/08
 * Purpose: a simple implementation of HashTable in C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*=================hash table start=========================================*/

#define HASH_TABLE_MAX_SIZE 9
typedef struct HashNode_Struct HashNode;

struct HashNode_Struct
{
    char *sKey;
    int nValue;
    HashNode *pNext;
};

HashNode *hashTable[HASH_TABLE_MAX_SIZE]; // hash table data strcutrue
int hash_table_size;                      // the number of key-value pairs in the hash table!

// initialize hash table
void hash_table_init()
{
    hash_table_size = 0;
    memset(hashTable, 0, sizeof(HashNode *) * HASH_TABLE_MAX_SIZE);
    printf("\n");
}

// string hash function
unsigned int hash_table_hash_str(const char *skey)
{
    const signed char *p = (const signed char *)skey;
    unsigned int h = *p;
    // int t = 0;
    if (h)
    {
        for (p += 1; *p != '\0'; ++p) {
            h = (h << 5) - h + *p;
            // t = (int)h;
            printf("h: %s, %u\n", p, h);
        }
    }
    return h;
}

// insert key-value into hash table
void hash_table_insert(const char *skey, int nvalue)
{
    if (hash_table_size >= HASH_TABLE_MAX_SIZE)
    {
        printf("out of hash table memory!\n");
        return;
    }

    unsigned int pos = hash_table_hash_str(skey) % HASH_TABLE_MAX_SIZE;
    printf("pos: %d\n", pos);
    HashNode *pHead = hashTable[pos];
    while (pHead)
    {
        if (strcmp(pHead->sKey, skey) == 0)
        {
            printf("%s already exists!\n", skey);
            return;
        }
        pHead = pHead->pNext;
    }

    HashNode *pNewNode = (HashNode *)malloc(sizeof(HashNode));
    memset(pNewNode, 0, sizeof(HashNode));
    size_t k = strlen(skey);
    pNewNode->sKey = (char *)malloc(sizeof(char) * (strlen(skey) + 1));
    strcpy(pNewNode->sKey, skey);
    pNewNode->nValue = nvalue;
    // hash 同值链表
    pNewNode->pNext = hashTable[pos];  // 原值附在新值后面
    hashTable[pos] = pNewNode;  // 新值接在链表头部

    hash_table_size++;
}

// remove key-value frome the hash table
void hash_table_remove(const char *skey)
{
    unsigned int pos = hash_table_hash_str(skey) % HASH_TABLE_MAX_SIZE;
    if (hashTable[pos])
    {
        HashNode *pHead = hashTable[pos];
        HashNode *pLast = NULL;
        HashNode *pRemove = NULL;
        while (pHead)
        {
            if (strcmp(skey, pHead->sKey) == 0)
            {
                pRemove = pHead;
                break;
            }
            pLast = pHead;
            pHead = pHead->pNext;
        }
        if (pRemove)
        {
            if (pLast)
                pLast->pNext = pRemove->pNext;
            else
                hashTable[pos] = NULL;

            free(pRemove->sKey);
            free(pRemove);
        }
    }
}

// lookup a key in the hash table
HashNode *hash_table_lookup(const char *skey)
{
    unsigned int pos = hash_table_hash_str(skey) % HASH_TABLE_MAX_SIZE;
    if (hashTable[pos])
    {
        HashNode *pHead = hashTable[pos];
        while (pHead)
        {
            if (strcmp(skey, pHead->sKey) == 0)
                return pHead;
            pHead = pHead->pNext;
        }
    }
    return NULL;
}

// print the content in the hash table
void hash_table_print()
{
    printf("===========content of hash table=================\n");
    int i;
    for (i = 0; i < HASH_TABLE_MAX_SIZE; ++i)
        if (hashTable[i])
        {
            HashNode *pHead = hashTable[i];
            printf("%d=>", i);
            while (pHead)
            {
                printf("%s:%d  ", pHead->sKey, pHead->nValue);
                pHead = pHead->pNext;
            }
            printf("\n");
        }
}

// free the memory of the hash table
void hash_table_release()
{
    int i;
    for (i = 0; i < HASH_TABLE_MAX_SIZE; ++i)
    {
        if (hashTable[i])
        {
            HashNode *pHead = hashTable[i];
            while (pHead)
            {
                HashNode *pTemp = pHead;
                pHead = pHead->pNext;
                if (pTemp)
                {
                    free(pTemp->sKey);
                    free(pTemp);
                }
            }
        }
    }
}

/* ===============================hash table end=========================*/

/* ============================test function ============================*/
#define MAX_STR_LEN 20
#define MIN_STR_LEN 10
void rand_str(char r[])
{
    int i;
    int len = MIN_STR_LEN + rand() % (MAX_STR_LEN - MIN_STR_LEN);
    for (i = 0; i < len - 1; ++i)
        r[i] = 'a' + rand() % ('z' - 'a');
    r[len - 1] = '\0';
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    hash_table_init();
    printf("insert testing.........\n");
    int n = 10;
    const char *key1 = "aaammd";
    const char *key2 = "xzzyym";
    const char *key3 = "cdcded中国";

    hash_table_insert(key1, 110);
    hash_table_insert(key2, 220);
    hash_table_insert(key3, 330);
    char str[MAX_STR_LEN + 1];
    printf("str: %s, %d\n", str, strlen(str));
    while (n--)
    {
        rand_str(str);
        printf("str: %s, %d\n", str, strlen(str));
        hash_table_insert(str, n);
    }
    hash_table_print();

    printf("\nlookup testing..........\n");
    HashNode *pNode = hash_table_lookup(key1);
    printf("lookup result:%d\n", pNode->nValue);
    pNode = hash_table_lookup(key2);
    printf("lookup result:%d\n", pNode->nValue);

    printf("\nremove testing..........\n");
    printf("before remove %s:\n", key3);
    hash_table_print();
    hash_table_remove(key3);
    printf("after remove:\n");
    hash_table_print();
    hash_table_release();

    system("pause");
    return 0;
}
