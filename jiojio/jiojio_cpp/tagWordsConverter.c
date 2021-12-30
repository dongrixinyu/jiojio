#include "tagWordsConverter.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

void tagWordsConverter(int a){ // const char* charList){//}, char *tags, long nodeNum) {

    PyObject* wordList = PyList_New(0);
    // npy_intp *nodeLength = PyArray_SHAPE(tags);
    // printf("node length: %ld", *nodeLength);
    // npy_intp *strideLength = PyArray_STRIDES(tags);
    // printf("stride length: %ld", *strideLength);
    long nodeNum = 1000;
    for(int i = 0; i < 10; i++) {
        printf("%ld", nodeNum);
    }


    // return wordList;
}

void main(){
    int node_num = 5;
    int tag_num = 2;

    float node_score[5][2] = {
        {0.2312, 1.2312}, {-0.241325, 0.8943},
        {-1.241325, -0.9943}, {3.241325, 1.8943},
         {-2.241325, 2.5943}};

    float edge_score[2][2] = {
        {0.5312, 0.6312}, {0.741325, -0.3943}};

    printf("%f\n", node_score[0][0]);
}
