#ifndef _CWSINFERENCE_H
#define _CWSINFERENCE_H

#include "Python.h"
#include "cwsPrediction.h"

void *init(
    int unigramSetHashTableMaxSize,
    PyObject *unigramPyList,
    int bigramSetHashTableMaxSize,
    PyObject *bigramPyList,
    int featureToIdxDictHashTableMaxSize,
    PyObject *featureToIdxPyList,
    PyObject *pyModelWeightList);

PyObject *cut(void *cwsPredictionObj, const wchar_t *text);

#endif
