#include "posFeatureExtractor.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

/**
 * @brief Append a current word length feature to the feature list
 *
 * @return PyObject*
 */
void addWordLengthFeature(Py_ssize_t curWordLength, PyObject *featureList)
{
    // 添加词长特征
    const wchar_t *wordLengthString = L"0123456789";
    const wchar_t *wordCurrentLength = L"wl";

    wchar_t *wordCurrentLengthFeature = malloc(3 * sizeof(wchar_t));
    wcsncpy(wordCurrentLengthFeature, wordCurrentLength, 2);

    if (curWordLength <= 9)
    {
        wcsncpy(wordCurrentLengthFeature + 2, wordLengthString + curWordLength, 1);
    }
    else
    {
        wcsncpy(wordCurrentLengthFeature + 2, wordLengthString, 1);
    }
    PyObject *tmpPyStr = PyUnicode_FromWideChar(wordCurrentLengthFeature, 3);
    int ret = PyList_Append(featureList, tmpPyStr);

    Py_DECREF(tmpPyStr);
    free(wordCurrentLengthFeature);
    wordCurrentLengthFeature = NULL;
}

/**
 * @brief Get the Node Feature Python object
 *
 * @param idx
 * @param wordList
 * @param nodeNum
 * @param singlePosWord: 单个词本身即决定词性的 set
 * @param unigram
 * @param bigram
 * @return PyObject*
 */
API PyObject *getPosNodeFeature(
    int idx, PyObject *wordList,
    PyObject *singlePosWord, PyObject *part, PyObject *unigram, PyObject *bigram)
{
    int ret = -1;

    PyObject *tmpPyStr;

    const wchar_t *wordCurrent = L"w";

    PyObject *featureList = PyList_New(0);
    PyObject *curWord = PyList_GetItem(wordList, idx);

    ret = PySet_Contains(singlePosWord, curWord);

    if (ret == 1)
    {
        // 当 wordList 列表中的获取的第 idx 个词汇在词典中，则直接返回结果

        PyObject *wordCurrentToken = PyUnicode_FromWideChar(wordCurrent, 1);
        tmpPyStr = PyUnicode_Concat(wordCurrentToken, curWord);
        Py_DECREF(wordCurrentToken);

        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);

        // Py_DECREF(curWord);
        return featureList;
    }

    // Py_DECREF(curWord);
    // Py_DECREF(tmpPyStr);
    // return featureList;

    // 获取词长
    Py_ssize_t curWordLength = PyUnicode_GET_LENGTH(curWord);

    // 添加前一词特征
    addWordLengthFeature(curWordLength, featureList);

    int before_word = 0; // 实际上是个指示flag

    PyObject *beforeLeftWordsList = PyList_New(0);
    PyObject *beforeRightWordsList = PyList_New(0);

    if (idx > 0)
    {
        // 获取前一词
        PyObject *beforeWord = PyList_GetItem(wordList, idx - 1);
        ret = PySet_Contains(unigram, beforeWord);
        Py_ssize_t beforeWordLength = PyUnicode_GET_LENGTH(beforeWord);

        if (ret == 1)
        {
            const wchar_t *wordBefore = L"v";

            PyObject *wordBeforeToken = PyUnicode_FromWideChar(wordBefore, 1);
            tmpPyStr = PyUnicode_Concat(wordBeforeToken, beforeWord);
            Py_DECREF(wordBeforeToken);
            ret = PyList_Append(featureList, tmpPyStr);
            Py_DECREF(tmpPyStr);

            before_word = 1;
        }
        else
        {
            int hasPart = 0;
            const wchar_t *partBeforeLeftToken = L"bl";
            const wchar_t *partBeforeRightToken = L"br";

            // 转换为 wchar_t*
            wchar_t *beforeWordTmp = PyUnicode_AsWideCharString(beforeWord, &beforeWordLength);

            for (int i = 1; i < beforeWordLength; i++)
            {
                // 左侧部分词特征
                wchar_t *beforeWordPartLeft = malloc(i * sizeof(wchar_t)); // 长度不定，如 “v中国”

                wcsncpy(beforeWordPartLeft, beforeWordTmp, i);

                PyObject *beforeWordPartLeftTmp = PyUnicode_FromWideChar(beforeWordPartLeft, i);
                // printf("cur left %ls\n", beforeWordPartLeft);
                ret = PySet_Contains(part, beforeWordPartLeftTmp);

                if (ret == 1)
                {
                    // 添加特征
                    wchar_t *beforeWordPartLeftFeature = malloc((i + 2) * sizeof(wchar_t));
                    wcsncpy(beforeWordPartLeftFeature, partBeforeLeftToken, 2);
                    wcsncpy(beforeWordPartLeftFeature + 2, beforeWordPartLeft, i);
                    tmpPyStr = PyUnicode_FromWideChar(beforeWordPartLeftFeature, i + 2);
                    ret = PyList_Append(featureList, tmpPyStr);

                    Py_DECREF(tmpPyStr);
                    free(beforeWordPartLeftFeature);
                    beforeWordPartLeftFeature = NULL;

                    // 添加部分词进入列表中
                    ret = PyList_Append(beforeLeftWordsList, beforeWordPartLeftTmp);

                    // flag 调整
                    hasPart = 1;
                }

                Py_DECREF(beforeWordPartLeftTmp);
                free(beforeWordPartLeft);
                beforeWordPartLeft = NULL;

                // 右侧部分词特征
                wchar_t *beforeWordPartRight = malloc(i * sizeof(wchar_t)); // 长度不定，如 “v中国”

                wcsncpy(beforeWordPartRight, beforeWordTmp + beforeWordLength - i, i);

                PyObject *beforeWordPartRightTmp = PyUnicode_FromWideChar(beforeWordPartRight, i);
                ret = PySet_Contains(part, beforeWordPartRightTmp);
                // printf("ret right: %d\n", ret);
                if (ret == 1)
                {
                    // 添加特征
                    wchar_t *beforeWordPartRightFeature = malloc((i + 2) * sizeof(wchar_t));
                    wcsncpy(beforeWordPartRightFeature, partBeforeRightToken, 2);
                    wcsncpy(beforeWordPartRightFeature + 2, beforeWordPartRight, i);
                    // printf("cur right feature %ls\n", beforeWordPartRightFeature);
                    tmpPyStr = PyUnicode_FromWideChar(beforeWordPartRightFeature, i + 2);
                    ret = PyList_Append(featureList, tmpPyStr);

                    Py_DECREF(tmpPyStr);
                    free(beforeWordPartRightFeature);
                    beforeWordPartRightFeature = NULL;

                    // 添加部分词进入列表中
                    ret = PyList_Append(beforeRightWordsList, beforeWordPartRightTmp);

                    // flag 调整
                    hasPart = 1;
                }

                Py_DECREF(beforeWordPartRightTmp);
                free(beforeWordPartRight);
                beforeWordPartRight = NULL;
            }

            if (hasPart == 0)
            {
                const wchar_t *wordBeforeUnknownToken = L"vk";
                tmpPyStr = PyUnicode_FromWideChar(wordBeforeUnknownToken, 2);
                ret = PyList_Append(featureList, tmpPyStr);

                Py_DECREF(tmpPyStr);
            }
        }
    }
    else
    {
        // 添加起始符特征
        const wchar_t *startFeature = L"[START]";

        tmpPyStr = PyUnicode_FromWideChar(startFeature, 7);
        ret = PyList_Append(featureList, tmpPyStr);
        Py_DECREF(tmpPyStr);
    }

    // Py_DECREF(curWord);

    Py_DECREF(beforeLeftWordsList);
    Py_DECREF(beforeRightWordsList);

    return featureList;
}
