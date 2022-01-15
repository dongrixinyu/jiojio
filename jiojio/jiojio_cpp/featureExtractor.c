#include "featureExtractor.h"

inline wchar_t *getSliceStr(wchar_t *text, int start, int length, int all_len)
{
    // printf("%d\t%d\t%d\n", start, length, all_len);
    // 空字符串
    wchar_t *emptyStr = malloc(1 * sizeof(wchar_t));
    memset(emptyStr, L'\0', 1 * sizeof(wchar_t));
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
PyObject *getNodeFeature(int idx, wchar_t *text, int nodeNum,
                         PySetObject *unigram, PySetObject *bigram)
{
    const wchar_t *startFeature = L"[START]";
    const wchar_t *endFeature = L"[END]";

    const wchar_t *delim = L".";
    // const wchar_t *emptyFeature = L"/";
    // const wchar_t *defaultFeature = L"$$";

    // 字符以 c 为中心，前后，z、a、b、c、d、e、f 依次扩展开
    const wchar_t *charCurrent = L"c";
    const wchar_t *charBefore = L"b";          // c-1.
    const wchar_t *charNext = L"d";            // c1.
    const wchar_t *charBefore2 = L"a";         // c-2.
    const wchar_t *charNext2 = L"e";           // c2.
    const wchar_t *charBefore3 = L"z";         // c-3.
    const wchar_t *charNext3 = L"f";           // c3.
    const wchar_t *charBeforeCurrent = L"bc";  // c-1c.
    const wchar_t *charBefore2Current = L"ac"; // c-2c.
    const wchar_t *charBefore3Current = L"zc"; // c-3c.
    const wchar_t *charCurrentNext = L"cd";    // cc1.
    const wchar_t *charCurrentNext2 = L"ce";   // cc2.
    const wchar_t *charCurrentNext3 = L"cf";   // cc3.
    // const wchar_t *charBefore21 = L"ab";       // c-2c-1.
    // const wchar_t *charNext12 = L"de";         // c1c2.

    const wchar_t *wordBefore = L"v";  // w-1.
    const wchar_t *wordNext = L"x";    // w1.
    const wchar_t *word2Left = L"wl";  // ww.l.
    const wchar_t *word2Right = L"wr"; // ww.r.

    int ret;
    wchar_t *curC = text + idx;        // 当前字符
    wchar_t *beforeC = text + idx - 1; // 前一个字符
    wchar_t *nextC = text + idx + 1;   // 后一个字符

    PyObject *featureList = PyList_New(0);
    // setlocale(LC_ALL, "en_US.UTF-8");

    // 添加当前字特征
    wchar_t *charCurrentFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “c佛”
    wcsncpy(charCurrentFeature, charCurrent, 1);
    wcsncpy(charCurrentFeature + 1, curC, 1);
    ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentFeature, 2));
    free(charCurrentFeature);

    if (idx > 0)
    {
        // 添加前一字特征
        wchar_t *charBeforeFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “b租”
        wcsncpy(charBeforeFeature, charBefore, 1);
        wcsncpy(charBeforeFeature + 1, beforeC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeFeature, 2));
        free(charBeforeFeature);

        // 添加当前字和前一字特征
        wchar_t *charBeforeCurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “bc佛.祖”
        wcsncpy(charBeforeCurrentFeature, charBeforeCurrent, 2);
        wcsncpy(charBeforeCurrentFeature + 2, beforeC, 1);
        wcsncpy(charBeforeCurrentFeature + 3, delim, 1);
        wcsncpy(charBeforeCurrentFeature + 4, curC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeCurrentFeature, 5));
        free(charBeforeCurrentFeature);
    }
    else
    {
        // 添加起始位特征
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(startFeature, 7));
    }

    if (idx < nodeNum - 1)
    {
        // 添加后一字特征
        wchar_t *charNextFeature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “d租”
        wcsncpy(charNextFeature, charNext, 1);
        wcsncpy(charNextFeature + 1, nextC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNextFeature, 2));
        free(charNextFeature);

        // 添加当前字和后一字特征
        wchar_t *charCurrentNextFeature = malloc(5 * sizeof(wchar_t)); // 默认为 2，如 “cd佛.祖”
        wcsncpy(charCurrentNextFeature, charCurrentNext, 2);
        wcsncpy(charCurrentNextFeature + 2, curC, 1);
        wcsncpy(charCurrentNextFeature + 3, delim, 1);
        wcsncpy(charCurrentNextFeature + 4, nextC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentNextFeature, 5));
        free(charCurrentNextFeature);
    }
    else
    {
        // 添加文本终止符特征
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
        free(charBeforeC2Feature);

        // 添加前第二字和当前字组合特征
        wchar_t *charBefore2CurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “sc佛.在”
        wcsncpy(charBefore2CurrentFeature, charBefore2Current, 2);
        wcsncpy(charBefore2CurrentFeature + 2, beforeC2, 1);
        wcsncpy(charBefore2CurrentFeature + 3, delim, 1);
        wcsncpy(charBefore2CurrentFeature + 4, curC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBefore2CurrentFeature, 5));
        free(charBefore2CurrentFeature);
    }

    if (idx < nodeNum - 2)
    {
        wchar_t *nextC2 = text + idx + 2;

        // 添加后第二字特征
        wchar_t *charNextC2Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “e租”
        wcsncpy(charNextC2Feature, charNext2, 1);
        wcsncpy(charNextC2Feature + 1, nextC2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNextC2Feature, 2));
        free(charNextC2Feature);

        // 添加当前字和后第二字组合特征
        wchar_t *charCurrentNext2Feature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “ce大.寺”
        wcsncpy(charCurrentNext2Feature, charCurrentNext2, 2);
        wcsncpy(charCurrentNext2Feature + 2, curC, 1);
        wcsncpy(charCurrentNext2Feature + 3, delim, 1);
        wcsncpy(charCurrentNext2Feature + 4, nextC2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentNext2Feature, 5));
        free(charCurrentNext2Feature);
    }

    if (idx > 2)
    {
        wchar_t *beforeC3 = text + idx - 3;

        // 添加前第三字特征
        wchar_t *charBeforeC3Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “z租”
        wcsncpy(charBeforeC3Feature, charBefore3, 1);
        wcsncpy(charBeforeC3Feature + 1, beforeC3, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBeforeC3Feature, 2));
        free(charBeforeC3Feature);

        // 添加前第三字和当前字组合特征
        wchar_t *charBefore3CurrentFeature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “zc佛.在”
        wcsncpy(charBefore3CurrentFeature, charBefore3Current, 2);
        wcsncpy(charBefore3CurrentFeature + 2, beforeC3, 1);
        wcsncpy(charBefore3CurrentFeature + 3, delim, 1);
        wcsncpy(charBefore3CurrentFeature + 4, curC, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charBefore3CurrentFeature, 5));
        free(charBefore3CurrentFeature);
    }

    if (idx < nodeNum - 3)
    {
        wchar_t *nextC3 = text + idx + 3;

        // 添加后第三字特征
        wchar_t *charNextC3Feature = malloc(2 * sizeof(wchar_t)); // 默认为 2，如 “f租”
        wcsncpy(charNextC3Feature, charNext3, 1);
        wcsncpy(charNextC3Feature + 1, nextC3, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charNextC3Feature, 2));
        free(charNextC3Feature);

        // 添加当前字和后第二字组合特征
        wchar_t *charCurrentNext3Feature = malloc(5 * sizeof(wchar_t)); // 默认为 5，如 “cf大.寺”
        wcsncpy(charCurrentNext3Feature, charCurrentNext3, 2);
        wcsncpy(charCurrentNext3Feature + 2, curC, 1);
        wcsncpy(charCurrentNext3Feature + 3, delim, 1);
        wcsncpy(charCurrentNext3Feature + 4, nextC3, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(charCurrentNext3Feature, 5));
        free(charCurrentNext3Feature);
    }

    int wordMax = 4;
    int wordMin = 2;
    wchar_t *wordLength = L"234";
    int preInFlag = 0; // 不仅指示是否进行双词匹配，也指示了匹配到的词汇的长度
    int preExFlag = 0;
    int postInFlag = 0;
    int postExFlag = 0;
    wchar_t *preIn;
    wchar_t *preEx;
    wchar_t *postIn;
    wchar_t *postEx;
    for (int l = wordMax; l > wordMin - 1; l--)
    {
        // printf("idx: %d\n", l);
        if (preInFlag == 0)
        {
            wchar_t *preInTmp = getSliceStr(text, idx - l + 1, l, nodeNum);

            if (wcslen(preInTmp) != 0)
            {
                ret = PySet_Contains(unigram, PyUnicode_FromWideChar(preInTmp, l));
                // printf("true:\t%d\t%ls\n", l, preInTmp);

                if (ret == 1)
                {
                    // 添加前一词特征
                    wchar_t *wordBeforeFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “v中国”
                    wcsncpy(wordBeforeFeature, wordBefore, 1);
                    wcsncpy(wordBeforeFeature + 1, preInTmp, l);
                    // printf("the string is right, %ls.\n", wordBeforeFeature);
                    ret = PyList_Append(featureList, PyUnicode_FromWideChar(wordBeforeFeature, l + 1));

                    // 记录该词
                    // preIn = malloc(l * sizeof(wchar_t));
                    // wcsncpy(preIn, preInTmp, l);
                    preIn = preInTmp;
                    free(wordBeforeFeature);

                    preInFlag = l;
                }
                free(preInTmp);
            }
        }

        if (postInFlag == 0)
        {
            wchar_t *postInTmp = getSliceStr(text, idx, l, nodeNum);

            if (wcslen(postInTmp) != 0)
            {
                ret = PySet_Contains(unigram, PyUnicode_FromWideChar(postInTmp, l));
                // printf("true postInTmp:\t%d\t%ls\n", l, postInTmp);

                if (ret == 1)
                {
                    // 添加后一词特征
                    wchar_t *wordNextFeature = malloc((1 + l) * sizeof(wchar_t)); // 长度不定，如 “x中国”
                    wcsncpy(wordNextFeature, wordNext, 1);
                    wcsncpy(wordNextFeature + 1, postInTmp, l);
                    ret = PyList_Append(featureList, PyUnicode_FromWideChar(wordNextFeature, l + 1));

                    // 记录该词
                    // postIn = malloc(l * sizeof(wchar_t));
                    // wcsncpy(postIn, postInTmp, l);
                    postIn = postInTmp;
                    free(wordNextFeature);

                    postInFlag = l;
                }
                free(postInTmp);
            }
        }

        if (preExFlag == 0)
        {
            wchar_t *preExTmp = getSliceStr(text, idx - l, l, nodeNum);

            if (wcslen(preExTmp) != 0)
            {
                ret = PySet_Contains(unigram, PyUnicode_FromWideChar(preExTmp, l));
                // printf("true preExTmp:\t%d\t%ls\n", l, preExTmp);

                if (ret == 1)
                {
                    // 记录该词
                    // preEx = malloc(l * sizeof(wchar_t));
                    // wcsncpy(preEx, preExTmp, l);
                    preEx = preExTmp;
                    preExFlag = l;
                }
                free(preExTmp);
            }
        }

        if (postExFlag == 0)
        {
            wchar_t *postExTmp = getSliceStr(text, idx + 1, l, nodeNum);

            if (wcslen(postExTmp) != 0)
            {
                ret = PySet_Contains(unigram, PyUnicode_FromWideChar(postExTmp, l));
                // printf("true postExTmp:\t%d\t%ls\n", l, postExTmp);

                if (ret == 1)
                {
                    // 记录该词
                    // postEx = malloc(l * sizeof(wchar_t));
                    // wcsncpy(postEx, postExTmp, l);
                    postEx = postExTmp;
                    postExFlag = l;
                }

                free(postExTmp);
            }
        }
    }
    // printf("## all length feature %d %d %d %d.\n",
    //        preExFlag, postInFlag, preInFlag, postExFlag);
    // 找到匹配的连续双词特征，此特征经过处理，仅保留具有歧义的连续双词
    if (preExFlag && postInFlag)
    {
        // printf("## add wl length feature %d %d.\n", preExFlag, postInFlag);
        wchar_t *bigramTmp = malloc((preExFlag + postInFlag + 1) * sizeof(wchar_t));
        wcsncpy(bigramTmp, preEx, preExFlag);
        wcsncpy(bigramTmp + preExFlag, delim, 1);
        wcsncpy(bigramTmp + 1 + preExFlag, postIn, postInFlag);

        ret = PySet_Contains(bigram, PyUnicode_FromWideChar(bigramTmp, preExFlag + postInFlag + 1));

        if (ret == 1)
        {
            wchar_t *bigramLeft = malloc((preExFlag + postInFlag + 3) * sizeof(wchar_t));
            wcsncpy(bigramLeft, word2Left, 2);
            wcsncpy(bigramLeft + 2, bigramTmp, preExFlag + postInFlag + 1);
            ret = PyList_Append(featureList, PyUnicode_FromWideChar(bigramLeft, preExFlag + postInFlag + 3));
            free(bigramLeft);
        }

        free(bigramTmp);

        // 添加词长特征
        wchar_t *bigramLeftLength = malloc(4 * sizeof(wchar_t));
        wcsncpy(bigramLeftLength, word2Left, 2);
        wcsncpy(bigramLeftLength + 2, wordLength + preExFlag - 2, 1);
        wcsncpy(bigramLeftLength + 3, wordLength + postInFlag - 2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(bigramLeftLength, 4));
        free(bigramLeftLength);
    }

    if ((preInFlag != 0) && (postExFlag != 0))
    {
        // printf("## add wr length feature %d %d.\n", preInFlag, postExFlag);
        wchar_t *bigramTmp = malloc((preInFlag + postExFlag + 1) * sizeof(wchar_t));
        wcsncpy(bigramTmp, preIn, preInFlag);
        wcsncpy(bigramTmp + preInFlag, delim, 1);
        wcsncpy(bigramTmp + 1 + preInFlag, postEx, postExFlag);

        ret = PySet_Contains(bigram, PyUnicode_FromWideChar(bigramTmp, preInFlag + postExFlag + 1));
        // printf("wr bigramTmp: %ls %d\n", bigramTmp, ret);
        if (ret == 1)
        {
            wchar_t *bigramRight = malloc((preInFlag + postExFlag + 3) * sizeof(wchar_t));
            wcsncpy(bigramRight, word2Right, 2);
            wcsncpy(bigramRight + 2, bigramTmp, preInFlag + postExFlag + 1);

            ret = PyList_Append(featureList, PyUnicode_FromWideChar(bigramRight, preInFlag + postExFlag + 3));
            // printf("wr bigramRight: %ls %d\n", bigramRight, ret);
            free(bigramRight);
        }

        free(bigramTmp);

        // 添加词长特征
        wchar_t *bigramRightLength = malloc(4 * sizeof(wchar_t));
        wcsncpy(bigramRightLength, word2Right, 2);
        wcsncpy(bigramRightLength + 2, wordLength + preInFlag - 2, 1);
        wcsncpy(bigramRightLength + 3, wordLength + postExFlag - 2, 1);
        ret = PyList_Append(featureList, PyUnicode_FromWideChar(bigramRightLength, 4));
        free(bigramRightLength);
    }
    // printf("## end of feature.\n");
    return featureList;
}
