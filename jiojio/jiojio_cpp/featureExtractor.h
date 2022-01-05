#include "Python.h"
#include <stdio.h>
// #include <stdbool.h>
#include <locale.h>

inline wchar_t *getSliceStr(wchar_t *text, int start, int length, int all_len);

PyObject *getNodeFeature(int idx, wchar_t *text, int nodeNum);
