#include "tagWordsConverter.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/** 执行计算 Python list 的 append 操作
 *
 */
inline void PyAppend(int wordLength, int start, PyObject *wordList, const wchar_t *charList)
{
    // wchar_t *clone = malloc(wordLength * sizeof(wchar_t));
    // wcsncpy(clone, charList + start, wordLength);
    // clone[wordLength] = L'\0'; // 字符串的结尾，会造成指针错乱无法释放

    // int ret = PyList_Append(wordList, PyUnicode_FromWideChar(clone, wordLength));
    int ret = PyList_Append(wordList, PyUnicode_FromWideChar(charList + start, wordLength));
    if (ret == -1)
        printf("Failed to append string to list.");
    // free(clone);
}

/** 从 tag 转为 words，默认 tags 参数的 strides 为 1 字节。

*/
PyObject *tagWordsConverter(const wchar_t *charList, char *tags, int nodeNum)
{

    PyObject *wordList = PyList_New(0);
    // npy_intp *nodeLength = PyArray_SHAPE(tags);
    // printf("node length: %ld", *nodeLength);

    // npy_intp *strideLength = PyArray_STRIDES(tags);
    // printf("stride length: %ld", *strideLength);
    // long nodeNum = 1000;
    // wchar_t *clone = malloc(3 * sizeof(wchar_t));
    // wchar_t *wcscpy(clone, charList);//, 2);
    // wcsncpy(clone, charList, 3);
    // clone[3] = '\0'; // 字符串的结尾
    // printf("clone: %ls\n", clone);
    // printf("tags:\t%d\n", *tags);

    int ret;
    int wordLength;
    int start = -1;
    // printf("nodeNum: %d\n", nodeNum);
    // printf("size of wchar_t: %ld\n", sizeof(wchar_t));

    for (int i = 0; i < nodeNum; i++)
    {
        // printf("%d\t%ls\t%d\t%d\n", i, charList + i, *(tags + i), tags[i]);
        if (tags[i] == 0)
        {
            if (i == 0)
            {
                start = i;
                if (i == nodeNum - 1) // 即文本中仅一个字
                {
                    ret = PyList_Append(wordList, PyUnicode_FromWideChar(charList, 1));
                    if (ret == -1)
                        printf("Failed to append string to list.");

                    break;
                };
                continue;
            }
            else if (i != nodeNum - 1)
            {
                if (start == -1)
                    continue;
                wordLength = i - start;
                PyAppend(wordLength, start, wordList, charList);
                start = i;
            }
            else
            {
                wordLength = i - start;
                PyAppend(wordLength, start, wordList, charList);

                ret = PyList_Append(wordList, PyUnicode_FromWideChar(charList + i, 1));
                if (ret == -1)
                    printf("Failed to append string to list.");
            }
        }
        else //if (tags[i] == 1)
        {
            if (i == 0)
            {
                start = i;
                continue;
            }
            else if (i != nodeNum - 1)
            {
                continue;
            }
            else
            {
                wordLength = i - start + 1;
                PyAppend(wordLength, start, wordList, charList);
            }
        }
    }

    return wordList;
}

void main()
{
    int node_num = 5;
    int tag_num = 2;

    float node_score[5][2] = {
        {0.2312, 1.2312}, {-0.241325, 0.8943}, {-1.241325, -0.9943}, {3.241325, 1.8943}, {-2.241325, 2.5943}};

    float edge_score[2][2] = {
        {0.5312, 0.6312}, {0.741325, -0.3943}};

    printf("%f\n", node_score[0][0]);
}
