#include "featureExtractor.h"

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

    const wchar_t *wordBefore = L"v";    // w-1.
    const wchar_t *wordNext = L"x";      // w1.
    const wchar_t *word2Left = L"wl";    // ww.l.
    const wchar_t *word2Right = L"wr中"; // ww.r.

    int ret;
    wchar_t *cur_c = text + idx;
    PyObject *featureList = PyList_New(0);
    typeof(word2Right) p_x = (word2Right);
    setlocale(LC_ALL, "en_US.UTF-8");
    printf("cur char: %ls\t%d\n", p_x, strlen(word2Right));
    ret = PyList_Append(featureList, PyUnicode_FromWideChar(word2Right, 3));

    return featureList;
}
