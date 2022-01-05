#include "featureExtractor.h"

inline wchar_t *getSliceStr(wchar_t *text, int start, int length, int all_len)
{
    // printf("%d\t%d\t%d\n", start, length, all_len);
    // 空字符串
    wchar_t *emptyStr = malloc(1 * sizeof(wchar_t));
    memset(emptyStr, L"\0", 1 * sizeof(wchar_t));
    emptyStr = L"\0";
    if (start < 0 || start >= all_len)
    {
        return emptyStr;
    }
    if (start + length > all_len)
    {
        return emptyStr;
    }

    // free(emptyStr);

    // 返回相应的子字符串
    wchar_t *resStr = malloc(length * sizeof(wchar_t));
    wcsncpy(resStr, text + start, length);
    return resStr;
}

/**
 * @brief Get the Node Feature Python object
 *
 * @param idx
 * @param text
 * @param nodeNum
 * @return PyObject*
 */
PyObject *getNodeFeature(int idx, wchar_t *text, int nodeNum)
{
    const wchar_t *startFeature = L"[START]";
    const wchar_t *endFeature = L"[END]";

    const wchar_t *delim = L".";
    const wchar_t *emptyFeature = L"/";
    const wchar_t *defaultFeature = L"$$";
    const wchar_t *charCurrent = L"c";
    const wchar_t *charBefore = L"b";         // c-1.
    const wchar_t *charNext = L"d";           // c1.
    const wchar_t *charBefore2 = L"a";        // c-2.
    const wchar_t *charNext2 = L"e";          // c2.
    const wchar_t *charBeforeCurrent = L"bc"; // c-1c.
    const wchar_t *charCurrentNext = L"cd";   // cc1.
    const wchar_t *charBefore21 = L"ab";      // c-2c-1.
    const wchar_t *charNext12 = L"de";        // c1c2.

    const wchar_t *wordBefore = L"v";  // w-1.
    const wchar_t *wordNext = L"x";    // w1.
    const wchar_t *word2Left = L"wl";  // ww.l.
    const wchar_t *word2Right = L"wr"; // ww.r.

    int ret;
    wchar_t *curC = text + idx;        // 当前字符
    wchar_t *beforeC = text + idx - 1; // 前一个字符
    wchar_t *nextC = text + idx + 1;   // 后一个字符

    PyObject *featureList = PyList_New(0);
    typeof(word2Right) p_x = (word2Right);
    setlocale(LC_ALL, "en_US.UTF-8");
    // printf("cur char: %ls\t%d\n", p_x, strlen(*word2Right));

    // 添加当前特征
    wchar_t *charCurrentFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “c佛”
    wcsncpy(charCurrentFeature, charCurrent, 1);
    wcsncpy(charCurrentFeature + 1, curC, 1);

    ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentFeature, 2));

    //
    if (idx > 0)
    {
        // 添加前一字特征
        wchar_t *charBeforeFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “b租”
        wcsncpy(charBeforeFeature, charBefore, 1);
        wcsncpy(charBeforeFeature + 1, beforeC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeFeature, 2));

        // 添加当前字和前一字特征
        wchar_t *charBeforeCurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “bc佛.祖”
        wcsncpy(charBeforeCurrentFeature, charBeforeCurrent, 2);
        wcsncpy(charBeforeCurrentFeature + 2, beforeC, 1);
        wcsncpy(charBeforeCurrentFeature + 3, delim, 1);
        wcsncpy(charBeforeCurrentFeature + 4, curC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeCurrentFeature, 5));
    }
    else
    {
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(startFeature, 7));
    }

    if (idx < nodeNum - 1)
    {

        // 添加后一字特征
        wchar_t *charNextFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “d租”
        wcsncpy(charNextFeature, charNext, 1);
        wcsncpy(charNextFeature + 1, nextC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNextFeature, 2));

        // 添加当前字和后一字特征
        wchar_t *charCurrentNextFeature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “cd佛.祖”
        wcsncpy(charCurrentNextFeature, charCurrentNext, 2);
        wcsncpy(charCurrentNextFeature + 2, curC, 1);
        wcsncpy(charCurrentNextFeature + 3, delim, 1);
        wcsncpy(charCurrentNextFeature + 4, nextC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentNextFeature, 5));
    }
    else
    {
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(endFeature, 5));
    }

    if (idx > 1)
    {
        wchar_t *beforeC2 = text + idx - 2;

        // 添加前第二字特征
        wchar_t *charBeforeC2Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “a租”
        wcsncpy(charBeforeC2Feature, charBefore2, 1);
        wcsncpy(charBeforeC2Feature + 1, beforeC2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeC2Feature, 2));

        // 添加前一字和前第二字组合特征
        wchar_t *charBefore21Feature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “bc佛.祖”
        wcsncpy(charBefore21Feature, charBefore21, 2);
        wcsncpy(charBefore21Feature + 2, beforeC2, 1);
        wcsncpy(charBefore21Feature + 3, delim, 1);
        wcsncpy(charBefore21Feature + 4, beforeC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBefore21Feature, 5));
    }

    if (idx < nodeNum - 2)
    {
        wchar_t *nextC2 = text + idx + 2;

        // 添加后第二字特征
        wchar_t *charNextC2Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “e租”
        wcsncpy(charNextC2Feature, charNext2, 1);
        wcsncpy(charNextC2Feature + 1, nextC2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNextC2Feature, 2));

        // 添加后一字和后第二字组合特征
        wchar_t *charNext12Feature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “de佛.祖”
        wcsncpy(charNext12Feature, charNext12, 2);
        wcsncpy(charNext12Feature + 2, nextC, 1);
        wcsncpy(charNext12Feature + 3, delim, 1);
        wcsncpy(charNext12Feature + 4, nextC2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNext12Feature, 5));
    }

    // printf("%d\n", hasWordFeature);

    int wordMax = 6;
    int wordMin = 2;
    wchar_t *preIn;
    wchar_t *preEx;
    wchar_t *postIn;
    wchar_t *postEx;
    for (int l = wordMax; l > wordMin - 1; l--)
    {
        printf("%d\n", l);

        wchar_t *preInTmp = getSliceStr(text, idx - l + 1, l, nodeNum);
        // if (preInTmp in unigram)
        //{

        if (strlen(preInTmp) != 0)
        {
            // 添加前一词特征
            printf("true:\t%d\t%s\n", l, preInTmp);
            wchar_t *wordBeforeFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “v中国”
            wcsncpy(wordBeforeFeature, wordBefore, 1);
            wcsncpy(wordBeforeFeature + 1, preInTmp, l);

            ret = PyList_Append(featureList, PyUnicode_FromWideChar(wordBeforeFeature, l + 1));
            free(preInTmp);
            free(wordBeforeFeature);

            // 记录 preIn 词汇
            preIn = malloc(l * sizeof(wchar_t));
            wcpncpy(preIn, preInTmp, l);
        }
        // break;
        // }
        // printf("false:\t%d\t%s\n", l, preInTmp);
    }

    for (int l = wordMax; l > wordMin - 1; l--)
    {
        printf("%d\n", l);
        wchar_t *postInTmp = getSliceStr(text, idx, l, nodeNum);
        if (strlen(postInTmp) != 0)
        {
            printf("true: %ls\n", postInTmp);
            // if (postInTmp in unigram)
            // {
            // 记录 postIn 词汇
            // postIn = malloc(l * sizeof(wchar_t));
            // wcpncpy(postIn, postInTmp, l);

            // 添加后一词特征
            wchar_t *wordNextFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “v中国”
            wcsncpy(wordNextFeature, wordNext, 1);
            wcsncpy(wordNextFeature + 1, postInTmp, l);
            ret = PyList_Append(featureList, PyUnicode_FromWideChar(wordNextFeature, l + 1));
            free(postIn);
            free(wordNextFeature);
            // break;
        }
    }

    return featureList;
}
