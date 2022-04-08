#ifndef _CWSFEATUREEXTRACTOR_H
#define _CWSFEATUREEXTRACTOR_H

#include "Python.h"
#include <stdio.h>
// #include <string>
// #include <stdbool.h>
// #include <locale.h>
// static const wchar_t *delim;

wchar_t *getSliceStr(wchar_t *text, int start, int length,
                     int all_len, wchar_t *emptyStr);

PyObject *getCwsNodeFeature(int idx, wchar_t *text, int nodeNum,
                            PyObject *unigram, PyObject *bigram);

#endif
