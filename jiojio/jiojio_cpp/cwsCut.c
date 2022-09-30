#include "cwsCut.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif


API PyObject *cwsCut(
    int idx, wchar_t *text, int nodeNum,
    PyObject *unigram, PyObject *bigram, PyObject *featureToIdx, PyObject *nodeWeight)
{

    struct Marker marker;  // 需要初始化

    // numpy 初始化
    for (int idx = 0; idx < nodeNum; idx++)
    {
        PyObject *nodeFeature = getCwsNodeFeature(
            idx, text, nodeNum, marker, unigram, bigram);

        PyObject *featureIdx = getFeatureIndex(nodeFeature, featureToIndex);

        // numpy 统计指标，返回 numpy.Array
        // PyObject *PyArray_Sum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
    }
}

int main() {
    //
}
