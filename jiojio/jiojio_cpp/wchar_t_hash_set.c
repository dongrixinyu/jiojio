#include "wchar_t_hash_set.h"

// initialize hash table
SetHashNode **set_hash_table_init(int hashTableMaxSize)
{
    SetHashNode **setHashTable = (SetHashNode **)malloc(sizeof(SetHashNode *) * hashTableMaxSize);
    memset(setHashTable, 0, sizeof(SetHashNode *) * hashTableMaxSize);

    return setHashTable;
}

// string hash function
unsigned int set_hash_table_hash_str(const wchar_t *skey)
{
    const signed char *p = (const signed char *)skey;
    unsigned int h = *p;
    int skeyLength = wcslen(skey) * sizeof(wchar_t);

    for (int i = 0; i < skeyLength; i++) {
        if (*(p + i) != '\0'){
            h = (h << 5) - h + *(p + i);
        }
    }

    return h;
}

// insert key into hash table
void set_hash_table_insert(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize)
{
    unsigned int pos = set_hash_table_hash_str(skey) % hashTableMaxSize;
    SetHashNode *pHead = *(setHashTable + pos);

    while (pHead)
    {
        if (wcscmp(pHead->sKey, skey) == 0)
        {
            printf("%ls already exists!\n", skey);
            return;
        }
        pHead = pHead->pNext;
    }

    SetHashNode *pNewNode = (SetHashNode *)malloc(sizeof(SetHashNode));
    memset(pNewNode, 0, sizeof(SetHashNode));
    pNewNode->sKey = (wchar_t *)malloc(sizeof(wchar_t) * (wcslen(skey) + 1));
    wcscpy(pNewNode->sKey, skey);

    // hash 同值链表
    pNewNode->pNext = *(setHashTable + pos); // 原值附在新值后面
    *(setHashTable + pos) = pNewNode; // 新值接在链表头部

}

// remove key frome the hash table
void set_hash_table_remove(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize)
{
    unsigned int pos = set_hash_table_hash_str(skey) % hashTableMaxSize;
    SetHashNode *temp = *(setHashTable + pos);
    if (temp)
    {
        SetHashNode *pHead = temp;
        SetHashNode *pLast = NULL;
        SetHashNode *pRemove = NULL;
        while (pHead)
        {
            if (wcscmp(skey, pHead->sKey) == 0)
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
                temp = NULL;

            free(pRemove->sKey);
            free(pRemove);
        }
    }
}

// lookup a key in the hash table
// 0 means failed to lookup and find, 1 means success.
// hash_table_hash_str costs 36% of total lookup time. others costs 64%.
int set_hash_table_lookup(SetHashNode **setHashTable, const wchar_t *skey, int hashTableMaxSize)
{
    unsigned int pos = set_hash_table_hash_str(skey) % hashTableMaxSize;
    SetHashNode *pHead = *(setHashTable + pos);

    while (pHead)
    {
        if (wcscmp(skey, pHead->sKey) == 0)
            return 1;

        pHead = pHead->pNext;
    }
    return 0;
}

double kl_divergence(int count_table[], int hashTableMaxSize, int hashTableSize)
{

    double pre_prob_log = log((double)hashTableMaxSize);
    double cross_entropy = 0.0, entropy = 0.0;

    for (int i = 0; i < hashTableMaxSize; i++)
    {
        double post_prob = (double)count_table[i] / hashTableSize;

        if (post_prob != 0) {
            cross_entropy += post_prob * pre_prob_log;
            entropy += post_prob * log(1 / post_prob);
        }
    }
    double kl_d = cross_entropy - entropy;
    printf("\tkl divergence: %f, diff_ratio: %f\n", kl_d, kl_d / entropy);
    return cross_entropy - entropy;  // kl-d value
}

float detect_empty_bucket(int count_table[], int hashTableMaxSize)
{
    int cnt = 0;
    for (int i = 0; i < hashTableMaxSize; i++)
    {
        if (count_table[i] == 0)
        {
            cnt++;
        }
    }

    float empty_ratio = (float)cnt / hashTableMaxSize;
    return empty_ratio;
}

void distribution_statistics(SetHashNode **setHashTable, int hashTableMaxSize, int hashTableSize)
{
    printf("===========content of hash table=================\n");
    int countTable[hashTableMaxSize];
    float averageNum = (float)hashTableSize / hashTableMaxSize;

    for (int i = 0; i < hashTableMaxSize; ++i)
    {
        if (*(setHashTable + i))
        {
            SetHashNode *pHead = *(setHashTable + i);
            int k = 0;
            while (pHead)
            {
                pHead = pHead->pNext;
                k++;
            }
            countTable[i] = k;
        }
        else
        {
            countTable[i] = 0;
        }
    }

    double kl_d = kl_divergence(countTable, hashTableMaxSize, hashTableSize);

    float empty_ratio = detect_empty_bucket(countTable, hashTableMaxSize);
    printf("\tempty bucket ratio: %f\n", empty_ratio);
}

// free the memory of the hash table
void set_hash_table_release(SetHashNode **setHashTable, int hashTableMaxSize)
{
    int i;
    for (i = 0; i < hashTableMaxSize; ++i)
    {
        if (*(setHashTable + i))
        {
            SetHashNode *pHead = *(setHashTable + i);
            while (pHead)
            {
                SetHashNode *pTemp = pHead;
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

void rand_widechar_str(wchar_t r[])
{
    int i;
    int max_str_len = 20;
    int min_str_len = 10;
    const wchar_t *baseLetters = L"abcdefghijklmnopqrstuvwxyz";
    int len = min_str_len + rand() % (max_str_len, min_str_len);
    for (i = 0; i < len - 1; ++i) {
        r[i] = baseLetters[rand() % ('z' - 'a')];
    }
    r[len - 1] = L'\0';
}

// int main(int argc, char **argv)
// {
//     setlocale(LC_ALL, "");
//     srand(time(NULL));

//     int setHashTableMaxSize = 10000;
//     SetHashNode **setHashTableObj = set_hash_table_init(setHashTableMaxSize);

//     // read_unigram
//     FILE *fp = NULL;
//     wchar_t buff[30];

//     clock_t start, end, look_up_start, look_up_end;
//     start = clock();
//     fp = fopen("/home/cuichengyu/github/jiojio/jiojio/models/default_cws_model/unigram.txt", "r");
//     int unigramCount = 0;
//     while (fgetws(buff, 30, (FILE *)fp))
//     {
//         int lens = wcslen(buff);
//         buff[lens-1] = L'\0';
//         // set_hash_table_insert(setHashTableObj, buff, setHashTableMaxSize);
//         // printf("=>: %ls, %d\n", buff, wcslen(buff));
//         unigramCount++;
//     }
//     fclose(fp);
//     end = clock();
//     printf("build hash table   time=%f\n", (double)(end - start) / CLOCKS_PER_SEC);

//     unsigned int qq;
//     double average_time[100];
//     for (int j = 0; j < 100; j++)
//     {
//         // empty run
//         start = clock();
//         fp = fopen("/home/cuichengyu/github/jiojio/jiojio/models/default_cws_model/unigram.txt", "r");
//         while (fgetws(buff, 30, (FILE *)fp))
//         {
//             int lens = wcslen(buff);
//             buff[lens - 1] = L'\0';
//         }
//         fclose(fp);
//         end = clock();
//         // printf("empty build        time=%f\n", (double)(end - start) / CLOCKS_PER_SEC);

//         // look up time
//         look_up_start = clock();
//         int res;
//         fp = fopen("/home/cuichengyu/github/jiojio/jiojio/models/default_cws_model/unigram.txt", "r");
//         while (fgetws(buff, 30, (FILE *)fp))
//         {
//             int lens = wcslen(buff);
//             buff[lens - 1] = L'\0';
//             // res = set_hash_table_lookup(setHashTableObj, buff, setHashTableMaxSize);
//             qq = set_hash_table_hash_str(buff);
//         }
//         fclose(fp);
//         look_up_end = clock();
//         // printf("look up hash table  time=%f\n", (double)(look_up_end - look_up_start) / CLOCKS_PER_SEC);
//         printf("pure look up        time=%f\n",
//                ((double)(look_up_end - look_up_start) - (double)(end - start)) / CLOCKS_PER_SEC);
//         average_time[j] = (double)(look_up_end - look_up_start) - (double)(end - start);
//     }

//     double average = 0;
//     for (int j = 0; j < 100; j++){
//         average += average_time[j];
//     }
//     printf("average time: %f\n", average / (100 * CLOCKS_PER_SEC));
//     distribution_statistics(setHashTableObj, setHashTableMaxSize, 89411);

//     // generate random wchar_t * string.
//     // wchar_t str[MAX_STR_LEN + 1];
//     // printf("str: %ls, %d\n", str, wcslen(str));
//     // rand_widechar_str(str);

//     set_hash_table_release(setHashTableObj, setHashTableMaxSize);

//     // system("pause");
//     return 0;
// }
