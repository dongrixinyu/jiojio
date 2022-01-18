#include "Python.h"
#include <stdio.h>
// #include <string>
// #include <stdbool.h>
#include <locale.h>

inline wchar_t *getSliceStr(wchar_t *text, int start, int length,
                            int all_len, wchar_t *emptyStr);

PyObject *getNodeFeature(int idx, wchar_t *text, int nodeNum,
                         PySetObject *unigram, PySetObject *bigram);
