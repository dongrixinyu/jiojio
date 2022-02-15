#ifndef _TAGWORDSCONVERTER_H
#define _TAGWORDSCONVERTER_H

#include <Python.h>
// #include "numpy/arrayobject.h"
// #include "numpy/ndarrayobject.h"
// #include "numpy/ndarraytypes.h"
// #include "numpy/arrayscalars.h"
// #include "numpy/ufuncobject.h"
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// # include <stdio.h>
void PyAppend(int wordLength, int start, PyObject *wordList, const wchar_t *charList);

PyObject *tagWordsConverter(const wchar_t *charList, char *tags, int nodeNum);

#endif
