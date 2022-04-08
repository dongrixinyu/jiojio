#ifndef _POSFEATUREEXTRACTOR_H
#define _POSFEATUREEXTRACTOR_H

#include "Python.h"
#include <stdio.h>
// #include <string>
// #include <stdbool.h>
// #include <locale.h>

// wchar_t *getSliceStr(wchar_t *text, int start, int length,
//                      int all_len, wchar_t *emptyStr);

void addWordLengthFeature(Py_ssize_t curWordLength, PyObject *featureList);

PyObject *getPosNodeFeature(
    int idx, PyObject *wordList,
    PyObject *singlePosWord, PyObject *part,
    PyObject *unigram, PyObject *bigram);

#endif
