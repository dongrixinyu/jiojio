// #pragma once

#ifndef _CWSPREDICTION_H
#define _CWSPREDICTION_H

#include <stdio.h>
#include "Python.h"
#include <wchar.h>
#include "wchar_t_hash_set.h"
#include "wchar_t_hash_dict.h"

// #include <locale.h>

typedef struct _CCwsPrediction CwsPrediction;
typedef struct _ConstLabels ConstLabels;

typedef int (*fptrInit)(CwsPrediction *, int, const char *, int, const char *);
typedef PyObject *(*fptrCut)(CwsPrediction *, wchar_t *);

typedef struct _ConstLabels
{
    const wchar_t *startFeature;
    const wchar_t *endFeature;
    const wchar_t *delim;

    // 字符以 c 为中心，前后，z、a、b、c、d、e、f 依次扩展开
    const wchar_t *charCurrent;
    const wchar_t *charBefore;         // c-1.
    const wchar_t *charNext;           // c1.
    const wchar_t *charBefore2;        // c-2.
    const wchar_t *charNext2;          // c2.
    const wchar_t *charBefore3;        // c-3.
    const wchar_t *charNext3;          // c3.
    const wchar_t *charBeforeCurrent;  // c-1c.
    const wchar_t *charBefore2Current; // c-2c.
    const wchar_t *charBefore3Current; // c-3c.
    const wchar_t *charCurrentNext;    // cc1.
    const wchar_t *charCurrentNext2;   // cc2.
    const wchar_t *charCurrentNext3;   // cc3.
    // const wchar_t *charBefore21 = L"ab";       // c-2c-1.
    // const wchar_t *charNext12 = L"de";         // c1c2.

    const wchar_t *wordBefore; // w-1.
    const wchar_t *wordNext;   // w1.
    const wchar_t *word2Left;  // ww.l.
    const wchar_t *word2Right; // ww.r.

    int wordMax;
    int wordMin;
    const wchar_t *wordLength;
} ConstLabels;

typedef struct _CCwsPrediction
{
    int unigramSetHashTableItemSize;
    int unigramSetHashTableMaxSize;
    SetHashNode **UnigramSetHashTable;

    int bigramSetHashTableItemSize;
    int bigramSetHashTableMaxSize;
    SetHashNode **BigramSetHashTable;

    int featureToIdxDictHashTableItemSize;
    int featureToIdxDictHashTableMaxSize;
    DictHashNode **featureToIdxDictHashTable;

    ConstLabels *constLabels;

    fptrInit _Init;
    fptrCut _Cut;

} CwsPrediction;

ConstLabels *newConstLabels();
CwsPrediction *newCwsPrediction();

int InitFile(CwsPrediction *cwsPredictionObj,
         int unigramSetHashTableMaxSize,
         const char *unigramFilePath,
         int bigramSetHashTableMaxSize,
         const char *bigramFilePath,
         int featureToIdxDictHashTableMaxSize,
         const char *featureToIdxFilePath);

int Init(
    CwsPrediction *cwsPredictionObj,
    int unigramSetHashTableMaxSize,
    PyObject *unigramPyList,
    int bigramSetHashTableMaxSize,
    PyObject *bigramPyList,
    int featureToIdxDictHashTableMaxSize,
    PyObject *featureToIdxPyList);

PyObject *Cut(CwsPrediction *cwsPredictionObj, const wchar_t *text);

wchar_t *getSliceStr(const wchar_t *text, int start, int length, int all_len, wchar_t *emptyStr);
wchar_t **getCwsNodeFeature(CwsPrediction *cwsPredictionObj,
                            int idx, const wchar_t *text, int nodeNum);

PyObject *getFeatureIndex(CwsPrediction *cwsPredictionObj,
                          wchar_t **featureList);

#endif
