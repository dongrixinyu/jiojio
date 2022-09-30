#ifndef _CWSCUT_H
#define _CWSCUT_H

#include "Python.h"
#include <stdio.h>

// #include "cwsFeatureExtractor.h"
// #include "cwsFeatureToIndex.h"


struct Marker {
    const wchar_t *startFeature;
    const wchar_t *endFeature;
    const wchar_t *delim;

    // 字符以 c 为中心，前后，z、a、b、c、d、e、f 依次扩展开
    const wchar_t *charCurrent;
    const wchar_t *charBefore;          // c-1.
    const wchar_t *charNext;            // c1.
    const wchar_t *charBefore2;         // c-2.
    const wchar_t *charNext2;           // c2.
    const wchar_t *charBefore3;         // c-3.
    const wchar_t *charNext3;           // c3.
    const wchar_t *charBeforeCurrent;  // c-1c.
    const wchar_t *charBefore2Current; // c-2c.
    const wchar_t *charBefore3Current; // c-3c.
    const wchar_t *charCurrentNext;    // cc1.
    const wchar_t *charCurrentNext2;   // cc2.
    const wchar_t *charCurrentNext3;   // cc3.

    const wchar_t *wordBefore;  // w-1.
    const wchar_t *wordNext;    // w1.
    const wchar_t *word2Left;   // ww.l.
    const wchar_t *word2Right;  // ww.r.
};

PyObject *getCwsNodeFeature(
    int idx, wchar_t *text, int nodeNum,
    struct Marker marker, PyObject *unigram, PyObject *bigram);

PyObject *getFeatureIndex(PyObject *nodeFeature, PyObject *featureToIndex);

PyObject *cwsCut(
    int idx, wchar_t *text, int nodeNum,
    PyObject *unigram, PyObject *bigram, PyObject *featureToIdx, PyObject *nodeWeight);

        // length = len(text)
        // all_features = []

        // # 每个节点的得分
        // # Y = np.empty((length, 2), dtype=np.float16)
        // for idx in range(length):

        //     node_features = self.get_node_features_c(
        //         idx, text, length, self.feature_extractor.unigram,
        //         self.feature_extractor.bigram)

        //     node_feature_idx = self.cws_feature2idx_c(
        //         node_features, self.feature_extractor.feature_to_idx)

        //     all_features.append(node_feature_idx)
        //     # Y[idx] = np.sum(node_weight[node_feature_idx], axis=0)

        // Y = get_log_Y_YY(all_features, self.model.node_weight, dtype=np.float16)

#endif
