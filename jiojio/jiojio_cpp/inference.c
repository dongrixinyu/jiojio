#include "inference.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

API PyObject *runViterbiDecode(
    float *node_score, float *edge_score, int node_num, int tag_num)
{
    printf("%f\t%f\n", node_score[1], node_score[9]);
    printf("finished!\n");
    PyObject *a = PyList_New(NULL);
    return a;
}

int main()
{
    int node_num = 5;
    int tag_num = 2;

    float node_score[5][2] = {
        {0.2312, 1.2312}, {-0.241325, 0.8943}, {-1.241325, -0.9943}, {3.241325, 1.8943}, {-2.241325, 2.5943}};

    float edge_score[2][2] = {
        {0.5312, 0.6312}, {0.741325, -0.3943}};
    // PyObject *featureList = PyList_New(NULL);

    PyObject *a = runViterbiDecode(
        *node_score, *edge_score, node_num, tag_num);
    printf("%f\n", node_score[0][0]);
}
