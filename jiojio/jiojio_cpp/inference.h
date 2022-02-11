#ifndef _INFERENCE_H
#define _INFERENCE_H

#include <Python.h>
// # include <stdio.h>
// # include <vector>

PyObject *runViterbiDecode(
    float *node_score, float *edge_score, int node_num, int tag_num);

#endif
