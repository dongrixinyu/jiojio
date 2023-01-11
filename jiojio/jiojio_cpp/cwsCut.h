#ifndef _CWSCUT_H
#define _CWSCUT_H

#include "Python.h"
#include <stdio.h>
// #include "numpy/ndarrayobject.h"
#include "cwsFeatureToIndex.h"
#include "cwsFeatureExtractor.h"

typedef const struct Marker {
    const wchar_t *startFeature;
    const wchar_t *endFeature;
    const wchar_t *delim;

    // 字符以 c 为中心，前后，z、a、b、c、d、e、f 依次扩展开
    const wchar_t *charCurrent;
    const wchar_t *charBefore;          // c-1.
    const wchar_t *charNext;            // c1.
    const wchar_t *charBefore2;         // c-2.
    const wchar_t *charNext2;           // c2.
    const wchar_t *charBefore3;         // c-3.
    const wchar_t *charNext3;           // c3.
    const wchar_t *charBeforeCurrent;  // c-1c.
    const wchar_t *charBefore2Current; // c-2c.
    const wchar_t *charBefore3Current; // c-3c.
    const wchar_t *charCurrentNext;    // cc1.
    const wchar_t *charCurrentNext2;   // cc2.
    const wchar_t *charCurrentNext3;   // cc3.

    const wchar_t *wordBefore;  // w-1.
    const wchar_t *wordNext;    // w1.
    const wchar_t *word2Left;   // ww.l.
    const wchar_t *word2Right;  // ww.r.
} Marker;

// wchar_t *getSliceStr(wchar_t *text, int start, int length,
                    //  int all_len, wchar_t *emptyStr);

PyObject *getCwsNodeFeatureStruct(
    int idx, wchar_t *text, int nodeNum,
    struct Marker marker, PyObject *unigram, PyObject *bigram);

// PyObject *getFeatureIndex(PyObject *nodeFeature, PyObject *featureToIndex);

PyObject *cwsCut(
    wchar_t *text, int nodeNum,
    PyObject *unigram, PyObject *bigram, PyObject *featureToIdx);

#endif
