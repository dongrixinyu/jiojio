// # ifndef INFERENCE_H
// # define INFERENCE_H

# include <Python.h>
// # include <stdio.h>
// # include <vector>

PyObject* runViterbiDecode(
    float* node_score, float* edge_score, int node_num, int tag_num);

class FeatureExtractor{
    public:
        PyObject* getNodeFeature()

    private:
        const char* emptyFeature;
        const char* defaultFeature;
        const char* delim;
        const char* charCurrent;
        const char* charBefore;
        const char* charNext;
        const char* charBefore2;
        const char* charNext2;

}

        self.char_before_current = 'bc'  # 'c-1c.'
        self.char_current_next = 'cd'  # 'cc1.'
        self.char_before_2_1 = 'ab'  # 'c-2c-1.'
        self.char_next_1_2 = 'de'  # 'c1c2.'

        self.word_before = 'v'  # 'w-1.'
        self.word_next = 'x'  # 'w1.'
        self.word_2_left = 'wl'  # 'ww.l.'
        self.word_2_right = 'wr'  # 'ww.r.'

        self.no_word = "**noWord"
