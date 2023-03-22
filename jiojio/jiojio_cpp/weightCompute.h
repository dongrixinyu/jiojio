#include <math.h>
#include "Python.h"
#include "numpy/arrayobject.h"

float **initModelWeight(PyObject *pyModelWeightList);

PyObject *computeNodeWeight(float **modelWeightObj, int **featureIdxListPtr, int textLength);
