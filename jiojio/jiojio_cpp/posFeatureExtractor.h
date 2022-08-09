#ifndef _POSFEATUREEXTRACTOR_H
#define _POSFEATUREEXTRACTOR_H

#include "Python.h"
#include <stdio.h>
// #include <string>
// #include <stdbool.h>
// #include <locale.h>
#define max(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

int getUnknownFeature(
    const wchar_t *flagToken, int flagTokenLength, PyObject *featureList);

int getUnigramFeatureWideChar(
    const wchar_t *flagToken, int flagTokenLength, wchar_t *word,
    Py_ssize_t wordLength, PyObject *featureList);

int getUnigramFeaturePythonString(
    const wchar_t *flagToken, int flagTokenLength, PyObject *word,
    Py_ssize_t wordLength, PyObject *featureList);

int getBigramFeature(
    const wchar_t *flagToken, int flagTokenLength, const wchar_t *mark,
    PyObject *firstWord, PyObject *secondWord,
    int firstWordLength, int secondWordLength,
    PyObject *featureList);

PyObject *getPosNodeFeature(
    int idx, PyObject *wordList, // int wordListLength,
    PyObject *part, PyObject *unigram, PyObject *chars);

#endif
