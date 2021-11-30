import re
import json
import jionlp as jio

results_list = list()
searched_pattern = re.compile(r'( \[([一-龥]+)\] )')
chinese_pattern = re.compile(r'^([一-龥]+)$')
file_path = '/home/ubuntu/datasets/word_segmentation/all.txt'
word_segmentation = list()
part_of_speech = list()
w_file_path = '/home/ubuntu/datasets/word_segmentation/pos.txt'
with open(w_file_path, 'w', encoding='utf-8') as fw:
    for idx, line in enumerate(jio.read_file_by_iter(file_path, strip=False)):
        # print(line)
        if line in ['undefined\n']:
            continue

        result = searched_pattern.findall(line)
        results_list.extend(result)
        line = line.replace(' [ [谁] ] ', '谁').replace(' [ [哪个] ] ', '哪个')
        line = searched_pattern.sub(r'\2', line)

        # print(line)
        line_list = line.split(' ')
        pos_list = list()
        for word_idx, seg in enumerate(line_list):
            if seg in ['', '\n']:
                continue
            if chinese_pattern.search(seg):
                if word_idx == len(line_list) - 2:
                    continue

            if len(seg.split('/')) != 2:
                print(seg)
                continue

            assert len(seg.split('/')) == 2, 'the `/` is wrong.'
            word, pos = seg.split('/')
            if word == '' or pos == '':
                # print(seg)
                if 'q/' in seg:
                    seg = seg.replace('q/', '/q')
                    word, pos = seg.split('/')
                    if word != '':
                        pos_list.append([word, pos])
                    continue
                # else:
                #     print(idx, seg)

            if word != '' and pos != '':
                pos_list.append([word, pos])
            else:
                print(idx, seg)

        fw.write(json.dumps(pos_list, ensure_ascii=False) + '\n')


# jio.write_file_by_line(word_segmentation, w_file_path)
print(set(results_list))
