#ifndef _CWSINFERENCE_H
#define _CWSINFERENCE_H

#include "Python.h"
#include "cwsPrediction.h"

void *new_cws_prediction();

int init(
    void *voidObj,
    int unigramSetHashTableMaxSize,
    PyObject *unigramPyList,
    int bigramSetHashTableMaxSize,
    PyObject *bigramPyList,
    int featureToIdxDictHashTableMaxSize,
    PyObject *featureToIdxPyList);

PyObject *cut(void *cwsPredictionObj, const wchar_t *text);

#endif
