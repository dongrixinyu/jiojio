#include "weightCompute.h"

float **initModelWeight(PyObject *pyModelWeightList) {

    Py_ssize_t weightLength = PyList_Size(pyModelWeightList);

    // malloc memory for initialization of model weight
    float **modelWeightObj = (float **)malloc(weightLength * sizeof(float *));

    for (int i = 0; i < weightLength; i++)
    {
        float *curWeightPair = malloc(2 * sizeof(float));
        PyObject *curPyWeightPair = PyList_GetItem(pyModelWeightList, i);

        PyObject *firstPyWeight = PyList_GetItem(curPyWeightPair, 0);
        PyObject *secondPyWeight = PyList_GetItem(curPyWeightPair, 1);

        // 数字强转可能带来问题
        float firstWeight = PyFloat_AsDouble(firstPyWeight);
        float secondWeight = PyFloat_AsDouble(secondPyWeight);

        *curWeightPair = firstWeight;
        *(curWeightPair + 1) = secondWeight;

        *(modelWeightObj + i) = curWeightPair;
    }

    return modelWeightObj;
}

/**
 * featureIdx: list of several idx list with size
 */
PyObject *computeNodeWeight(float** modelWeightObj, int **featureIdxListPtr, int textLength) {

    int dims[2] = {textLength, 2};
    int ndim = 2;
    import_array();
    printf("type num: %d\n", NPY_FLOAT32);
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_FLOAT32); // designate np.float32
    if (descr == NULL)
    {
        printf("error occured!\n");
    }

    PyObject *arr = PyArray_Empty(ndim, dims, descr, 0);
    PyArrayObject *arr_ptr = (PyArrayObject *)arr;
    float *data = (float *)PyArray_DATA(arr_ptr);

    int ret;

    for (int i = 0; i < textLength; i++) {
        float *curfeatureIdxList = *(featureIdxListPtr + i);

        float *sumNum = malloc(sizeof(float) * 2);
        memset(sumNum, 0.0, sizeof(float));

        while (curfeatureIdxList)
        {
            float curFeatureIdx = *curfeatureIdxList;

            *sumNum += **(modelWeightObj + curFeatureIdx);
            *(sumNum + 1) += *(*(modelWeightObj + curFeatureIdx) + 1);

            curfeatureIdxList++;
        }

        PyObject *firstSum = PyFloat_FromDouble((double)(*sumNum));
        PyObject *secondSum = PyFloat_FromDouble((double)(*(sumNum + 1)));

        PyArray_SETITEM(arr_ptr, data + 2 * i, firstSum);
        PyArray_SETITEM(arr_ptr, data + 2 * i + 1, secondSum);

    }

    return arr;
}
