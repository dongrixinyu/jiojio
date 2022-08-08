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

int getUnknownFeatrue(
    const wchar_t *flagToken, int flagTokenLength, PyObject *featureList);

int getUnigramFeatrueWideChar(
    const wchar_t *flagToken, int flagTokenLength, wchar_t *word,
    Py_ssize_t wordLength, PyObject *featureList);

int getUnigramFeatruePythonString(
    const wchar_t *flagToken, int flagTokenLength, PyObject *word,
    PyObject *featureList);

int getBigramFeature(
    const wchar_t *flagToken, int flagTokenLength, const wchar_t *mark,
    PyObject *firstWord, PyObject *secondWord,
    PyObject *featureList);

PyObject *getPosNodeFeature(
    int idx, PyObject *wordList,
    PyObject *part, PyObject *unigram, PyObject *chars);

#endif
