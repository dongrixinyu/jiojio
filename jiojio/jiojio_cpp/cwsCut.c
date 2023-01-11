#include "cwsCut.h"
// #include "cwsFeatureToIndex.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif


PyObject *getCwsNodeFeatureStruct(
    int idx, wchar_t *text, int nodeNum,
    struct Marker marker, PyObject *unigram, PyObject *bigram)
{
    wchar_t *emptyStr = malloc(sizeof(wchar_t));
    memset(emptyStr, L'\0', sizeof(wchar_t));

    const int wordMax = 4;
    const int wordMin = 2;
    const wchar_t *wordLength = L"234";

    int ret = -1;
    wchar_t *curC = text + idx;        // 当前字符
    wchar_t *beforeC = text + idx - 1; // 前一个字符
    wchar_t *nextC = text + idx + 1;   // 后一个字符

    PyObject *featureList = PyList_New(0);
    PyObject *tmpPyStr;
    // setlocale(LC_ALL, "en_US.UTF-8");

    // 添加当前字特征
    wchar_t *charCurrentFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “c佛”
    wcsncpy(charCurrentFeature, marker.charCurrent, 1);
    wcsncpy(charCurrentFeature + 1, curC, 1);
    tmpPyStr = PyUnicode_FromWideChar(charCurrentFeature, 2);
    ret = PyList_Append(featureList, tmpPyStr);
    Py_DECREF(tmpPyStr);
    free(charCurrentFeature);
    charCurrentFeature = NULL;

    if (idx > 0)
    {
        // 添加前一字特征
        wchar_t *charBeforeFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “b租”
        wcsncpy(charBeforeFeature, marker.charBefore, 1);
        wcsncpy(charBeforeFeature + 1, beforeC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBeforeFeature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBeforeFeature);
        charBeforeFeature = NULL;

        // 添加当前字和前一字特征
        wchar_t *charBeforeCurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “bc佛.祖”
        wcsncpy(charBeforeCurrentFeature, marker.charBeforeCurrent, 2);
        wcsncpy(charBeforeCurrentFeature + 2, beforeC, 1);
        wcsncpy(charBeforeCurrentFeature + 3, marker.delim, 1);
        wcsncpy(charBeforeCurrentFeature + 4, curC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBeforeCurrentFeature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBeforeCurrentFeature);
        charBeforeCurrentFeature = NULL;
    }
    else
    {
        // 添加起始位特征
        tmpPyStr = PyUnicode_FromWideChar(marker.startFeature, 7);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
    }

    if (idx < nodeNum - 1)
    {
        // 添加后一字特征
        wchar_t *charNextFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “d租”
        wcsncpy(charNextFeature, marker.charNext, 1);
        wcsncpy(charNextFeature + 1, nextC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charNextFeature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charNextFeature);
        charNextFeature = NULL;

        // 添加当前字和后一字特征
        wchar_t *charCurrentNextFeature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “cd佛.祖”
        wcsncpy(charCurrentNextFeature, marker.charCurrentNext, 2);
        wcsncpy(charCurrentNextFeature + 2, curC, 1);
        wcsncpy(charCurrentNextFeature + 3, marker.delim, 1);
        wcsncpy(charCurrentNextFeature + 4, nextC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charCurrentNextFeature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charCurrentNextFeature);
        charCurrentNextFeature = NULL;
    }
    else
    {
        // 添加文本终止符特征
        tmpPyStr = PyUnicode_FromWideChar(marker.endFeature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
    }

    if (idx > 1)
    {
        wchar_t *beforeC2 = text + idx - 2;

        // 添加前第二字特征
        wchar_t *charBeforeC2Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “a租”
        wcsncpy(charBeforeC2Feature, marker.charBefore2, 1);
        wcsncpy(charBeforeC2Feature + 1, beforeC2, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBeforeC2Feature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBeforeC2Feature);
        charBeforeC2Feature = NULL;

        // 添加前第二字和当前字组合特征
        wchar_t *charBefore2CurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “sc佛.在”
        wcsncpy(charBefore2CurrentFeature, marker.charBefore2Current, 2);
        wcsncpy(charBefore2CurrentFeature + 2, beforeC2, 1);
        wcsncpy(charBefore2CurrentFeature + 3, marker.delim, 1);
        wcsncpy(charBefore2CurrentFeature + 4, curC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBefore2CurrentFeature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBefore2CurrentFeature);
        charBefore2CurrentFeature = NULL;
    }

    if (idx < nodeNum - 2)
    {
        wchar_t *nextC2 = text + idx + 2;

        // 添加后第二字特征
        wchar_t *charNextC2Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “e租”
        wcsncpy(charNextC2Feature, marker.charNext2, 1);
        wcsncpy(charNextC2Feature + 1, nextC2, 1);
        tmpPyStr = PyUnicode_FromWideChar(charNextC2Feature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charNextC2Feature);
        charNextC2Feature = NULL;

        // 添加当前字和后第二字组合特征
        wchar_t *charCurrentNext2Feature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “ce大.寺”
        wcsncpy(charCurrentNext2Feature, marker.charCurrentNext2, 2);
        wcsncpy(charCurrentNext2Feature + 2, curC, 1);
        wcsncpy(charCurrentNext2Feature + 3, marker.delim, 1);
        wcsncpy(charCurrentNext2Feature + 4, nextC2, 1);
        tmpPyStr = PyUnicode_FromWideChar(charCurrentNext2Feature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charCurrentNext2Feature);
        charCurrentNext2Feature = NULL;
    }

    if (idx > 2)
    {
        wchar_t *beforeC3 = text + idx - 3;

        // 添加前第三字特征
        wchar_t *charBeforeC3Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “z租”
        wcsncpy(charBeforeC3Feature, marker.charBefore3, 1);
        wcsncpy(charBeforeC3Feature + 1, beforeC3, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBeforeC3Feature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBeforeC3Feature);
        charBeforeC3Feature = NULL;

        // 添加前第三字和当前字组合特征
        wchar_t *charBefore3CurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “zc佛.在”
        wcsncpy(charBefore3CurrentFeature, marker.charBefore3Current, 2);
        wcsncpy(charBefore3CurrentFeature + 2, beforeC3, 1);
        wcsncpy(charBefore3CurrentFeature + 3, marker.delim, 1);
        wcsncpy(charBefore3CurrentFeature + 4, curC, 1);
        tmpPyStr = PyUnicode_FromWideChar(charBefore3CurrentFeature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charBefore3CurrentFeature);
        charBefore3CurrentFeature = NULL;
    }

    if (idx < nodeNum - 3)
    {
        wchar_t *nextC3 = text + idx + 3;

        // 添加后第三字特征
        wchar_t *charNextC3Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “f租”
        wcsncpy(charNextC3Feature, marker.charNext3, 1);
        wcsncpy(charNextC3Feature + 1, nextC3, 1);
        tmpPyStr = PyUnicode_FromWideChar(charNextC3Feature, 2);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charNextC3Feature);
        charNextC3Feature = NULL;

        // 添加当前字和后第二字组合特征
        wchar_t *charCurrentNext3Feature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “cf大.寺”
        wcsncpy(charCurrentNext3Feature, marker.charCurrentNext3, 2);
        wcsncpy(charCurrentNext3Feature + 2, curC, 1);
        wcsncpy(charCurrentNext3Feature + 3, marker.delim, 1);
        wcsncpy(charCurrentNext3Feature + 4, nextC3, 1);
        tmpPyStr = PyUnicode_FromWideChar(charCurrentNext3Feature, 5);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(charCurrentNext3Feature);
        charCurrentNext3Feature = NULL;
    }

    int preInFlag = 0; // 不仅指示是否进行双词匹配，也指示了匹配到的词汇的长度
    int preExFlag = 0;
    int postInFlag = 0;
    int postExFlag = 0;
    wchar_t *preIn = NULL;
    wchar_t *preEx = NULL;
    wchar_t *postIn = NULL;
    wchar_t *postEx = NULL;
    for (int l = wordMax; l > wordMin - 1; l--)
    {
        if (preInFlag == 0)
        {
            wchar_t *preInTmp = getSliceStr(text, idx - l + 1, l, nodeNum, emptyStr);
            if (wcscmp(preInTmp, emptyStr) != 0)
            {
                tmpPyStr = PyUnicode_FromWideChar(preInTmp, l);
                ret = PySet_Contains(unigram, tmpPyStr);
                Py_DECREF(tmpPyStr);

                if (ret == 1)
                {
                    // 添加前一词特征
                    wchar_t *wordBeforeFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “v中国”
                    wcsncpy(wordBeforeFeature, marker.wordBefore, 1);
                    wcsncpy(wordBeforeFeature + 1, preInTmp, l);

                    tmpPyStr = PyUnicode_FromWideChar(wordBeforeFeature, l + 1);
                    ret = PyList_Append(featureList, tmpPyStr);
                    Py_DECREF(tmpPyStr);
                    free(wordBeforeFeature);
                    wordBeforeFeature = NULL;
                    // 记录该词
                    preIn = malloc(l * sizeof(wchar_t));
                    wcsncpy(preIn, preInTmp, l);
                    // preIn = preInTmp;
                    preInFlag = l;
                }
                // if (preInFlag == 0)
                free(preInTmp);
            }
            preInTmp = NULL;
        }

        if (postInFlag == 0)
        {
            wchar_t *postInTmp = getSliceStr(text, idx, l, nodeNum, emptyStr);

            if (wcscmp(postInTmp, emptyStr) != 0)
            {
                tmpPyStr = PyUnicode_FromWideChar(postInTmp, l);
                ret = PySet_Contains(unigram, tmpPyStr);
                Py_DECREF(tmpPyStr);
                if (ret == 1)
                {
                    // 添加后一词特征
                    wchar_t *wordNextFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “x中国”
                    wcsncpy(wordNextFeature, marker.wordNext, 1);
                    wcsncpy(wordNextFeature + 1, postInTmp, l);
                    tmpPyStr = PyUnicode_FromWideChar(wordNextFeature, l + 1);
                    ret = PyList_Append(featureList, tmpPyStr);
                    Py_DECREF(tmpPyStr);
                    free(wordNextFeature);

                    // 记录该词
                    postIn = malloc(l * sizeof(wchar_t));
                    wcsncpy(postIn, postInTmp, l);
                    // postIn = postInTmp;
                    postInFlag = l;
                }
                // if (postInFlag == 0)
                free(postInTmp);
            }
            postInTmp = NULL;
        }

        if (preExFlag == 0)
        {
            wchar_t *preExTmp = getSliceStr(text, idx - l, l, nodeNum, emptyStr);
            if (wcscmp(preExTmp, emptyStr) != 0)
            {
                tmpPyStr = PyUnicode_FromWideChar(preExTmp, l);
                ret = PySet_Contains(unigram, tmpPyStr);
                Py_DECREF(tmpPyStr);
                if (ret == 1)
                {
                    // 记录该词
                    preEx = malloc(l * sizeof(wchar_t));
                    wcsncpy(preEx, preExTmp, l);
                    // preEx = preExTmp;
                    preExFlag = l;
                }
                // if (preExFlag == 0)
                free(preExTmp);
            }
            preExTmp = NULL;
        }

        if (postExFlag == 0)
        {
            wchar_t *postExTmp = getSliceStr(text, idx + 1, l, nodeNum, emptyStr);
            if (wcscmp(postExTmp, emptyStr) != 0)
            {
                tmpPyStr = PyUnicode_FromWideChar(postExTmp, l);
                ret = PySet_Contains(unigram, tmpPyStr);
                Py_DECREF(tmpPyStr);
                if (ret == 1)
                {
                    // 记录该词
                    postEx = malloc(l * sizeof(wchar_t));
                    wcsncpy(postEx, postExTmp, l);
                    // postEx = postExTmp;
                    postExFlag = l;
                }

                // if (postExTmp == 0)
                free(postExTmp);
            }
            postExTmp = NULL;
        }
    }

    // 找到匹配的连续双词特征，此特征经过处理，仅保留具有歧义的连续双词
    if (preExFlag && postInFlag)
    {
        // printf("## add wl length feature %d %d.\n", preExFlag, postInFlag);
        wchar_t *bigramTmp = malloc((preExFlag + postInFlag + 1) * sizeof(wchar_t));
        wcsncpy(bigramTmp, preEx, preExFlag);
        wcsncpy(bigramTmp + preExFlag, marker.delim, 1);
        wcsncpy(bigramTmp + 1 + preExFlag, postIn, postInFlag);

        tmpPyStr = PyUnicode_FromWideChar(bigramTmp, preExFlag + postInFlag + 1);
        ret = PySet_Contains(bigram, tmpPyStr);
        Py_DECREF(tmpPyStr);

        if (ret == 1)
        {
            wchar_t *bigramLeft = malloc((preExFlag + postInFlag + 3) * sizeof(wchar_t));
            wcsncpy(bigramLeft, marker.word2Left, 2);
            wcsncpy(bigramLeft + 2, bigramTmp, preExFlag + postInFlag + 1);

            tmpPyStr = PyUnicode_FromWideChar(bigramLeft, preExFlag + postInFlag + 3);
            ret = PyList_Append(featureList, tmpPyStr);
            Py_DECREF(tmpPyStr);
            free(bigramLeft);
            bigramLeft = NULL;
        }

        free(bigramTmp);
        bigramTmp = NULL;

        // 添加词长特征
        wchar_t *bigramLeftLength = malloc(4 * sizeof(wchar_t));
        wcsncpy(bigramLeftLength, marker.word2Left, 2);
        wcsncpy(bigramLeftLength + 2, wordLength + preExFlag - 2, 1);
        wcsncpy(bigramLeftLength + 3, wordLength + postInFlag - 2, 1);
        tmpPyStr = PyUnicode_FromWideChar(bigramLeftLength, 4);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(bigramLeftLength);
        bigramLeftLength = NULL;
    }

    if ((preInFlag != 0) && (postExFlag != 0))
    {
        // printf("## add wr length feature %d %d.\n", preInFlag, postExFlag);
        wchar_t *bigramTmp = malloc((preInFlag + postExFlag + 1) * sizeof(wchar_t));
        wcsncpy(bigramTmp, preIn, preInFlag);
        wcsncpy(bigramTmp + preInFlag, marker.delim, 1);
        wcsncpy(bigramTmp + 1 + preInFlag, postEx, postExFlag);

        tmpPyStr = PyUnicode_FromWideChar(bigramTmp, preInFlag + postExFlag + 1);
        ret = PySet_Contains(bigram, tmpPyStr);
        Py_DECREF(tmpPyStr);
        // printf("wr bigramTmp: %ls %d\n", bigramTmp, ret);
        if (ret == 1)
        {
            wchar_t *bigramRight = malloc((preInFlag + postExFlag + 3) * sizeof(wchar_t));
            wcsncpy(bigramRight, marker.word2Right, 2);
            wcsncpy(bigramRight + 2, bigramTmp, preInFlag + postExFlag + 1);

            tmpPyStr = PyUnicode_FromWideChar(bigramRight, preInFlag + postExFlag + 3);
            ret = PyList_Append(featureList, tmpPyStr);
            Py_DECREF(tmpPyStr);
            free(bigramRight);
            bigramRight = NULL;
        }

        free(bigramTmp);
        bigramTmp = NULL;

        // 添加词长特征
        wchar_t *bigramRightLength = malloc(4 * sizeof(wchar_t));
        wcsncpy(bigramRightLength, marker.word2Right, 2);
        wcsncpy(bigramRightLength + 2, wordLength + preInFlag - 2, 1);
        wcsncpy(bigramRightLength + 3, wordLength + postExFlag - 2, 1);

        tmpPyStr = PyUnicode_FromWideChar(bigramRightLength, 4);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
        free(bigramRightLength);
        bigramRightLength = NULL;
    }

    if (preIn != NULL)
    {
        free(preIn);
        preIn = NULL;
    }
    if (preEx != NULL)
    {
        free(preEx);
        preEx = NULL;
    }
    if (postIn != NULL)
    {
        free(postIn);
        postIn = NULL;
    }
    if (postEx != NULL)
    {
        free(postEx);
        postEx = NULL;
    }
    free(emptyStr);
    emptyStr = NULL;

    return featureList;
}

API PyObject *cwsCut(
    wchar_t *text, int nodeNum,
    PyObject *unigram, PyObject *bigram, PyObject *featureToIdx)
{

    // struct Marker marker = {
    //     .startFeature = L"[START]",
    //     .endFeature = L"[END]",
    //     .delim = L".",

    //     // 字符以 c 为中心，前后，z、a、b、c、d、e、f 依次扩展开
    //     .charCurrent = L"c",
    //     .charBefore = L"b",          // c-1.
    //     .charNext = L"d",            // c1.
    //     .charBefore2 = L"a",         // c-2.
    //     .charNext2 = L"e",           // c2.
    //     .charBefore3 = L"z",         // c-3.
    //     .charNext3 = L"f",           // c3.
    //     .charBeforeCurrent = L"bc",  // c-1c.
    //     .charBefore2Current = L"ac", // c-2c.
    //     .charBefore3Current = L"zc", // c-3c.
    //     .charCurrentNext = L"cd",    // cc1.
    //     .charCurrentNext2 = L"ce",   // cc2.
    //     .charCurrentNext3 = L"cf",   // cc3.

    //     .wordBefore = L"v",  // w-1.
    //     .wordNext = L"x",    // w1.
    //     .word2Left = L"wl",  // ww.l.
    //     .word2Right = L"wr" // ww.r.
    // };

    PyObject *allIndexList = PyList_New(0);
    // numpy 初始化
    for (int idx = 0; idx < nodeNum; idx++)
    {
        PyObject *nodeFeature = getCwsNodeFeature(
            idx, text, nodeNum, unigram, bigram);

        PyObject *featureIdx = getFeatureIndex(nodeFeature, featureToIdx);

        int ret = PyList_Append(allIndexList, featureIdx);
        // numpy 统计指标，返回 numpy.Array
        // PyObject *PyArray_Sum(PyArrayObject *self, int axis, int rtype, PyArrayObject *out)
    }

    return allIndexList;
}

int main() {
    //
}
