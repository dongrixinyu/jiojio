#include "cwsInterface.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

void *new_cws_prediction() {
    CwsPrediction *cwsPredictionObj = newCwsPrediction();
    printf("# res!!!!\n");

    return cwsPredictionObj;
    // void *interface_p = (void *)cwsPredictionObj;

    // return interface_p;
}

int init(void *voidObj,
         int unigramSetHashTableMaxSize,
         PyObject *unigramPyList,
         int bigramSetHashTableMaxSize,
         PyObject *bigramPyList,
         int featureToIdxDictHashTableMaxSize,
         PyObject *featureToIdxPyList) {
    printf("###  C start allocate memory!");
    if (voidObj == NULL) {
        printf("# empty void Pointer. \n");
        return 0;
    }

    CwsPrediction *cwsPredictionObj = (CwsPrediction *)voidObj;
    int ret = Init(cwsPredictionObj,
                   unigramSetHashTableMaxSize,
                   unigramPyList,
                   bigramSetHashTableMaxSize,
                   bigramPyList,
                   featureToIdxDictHashTableMaxSize,
                   featureToIdxPyList);

    return ret;
}

PyObject *cut(void *voidObj, const wchar_t *text) {
    printf("###!!!\n");
    CwsPrediction *cwsPredictionObj = (CwsPrediction *)voidObj;
    PyObject *featureIdx = Cut(cwsPredictionObj, text);

    return featureIdx;
}
