# -*- coding=utf-8 -*-

import os
import pdb
import random
import hashlib
import numpy as np

import jionlp as jio

from jiojio.trie_tree import TrieTree


file_path = '/home/cuichengyu/dataset/cws.txt'
# file_path = '/home/cuichengyu/dataset/train_cws.txt1'

keys_dict = dict()
trie_tree_obj = TrieTree()


duplicate_res_list = list()
md5_length = 10
for idx, line in enumerate(jio.read_file_by_iter(
    file_path, auto_loads_json=False, strip=True)):
    key = hashlib.md5(line.encode('utf-8')).hexdigest()[:md5_length]

    _, res = trie_tree_obj.search(key)

    if res is None:
        trie_tree_obj.add_node(key, str(idx))
        duplicate_res_list.append(line)
    else:
        # 匹配到重复，丢弃该数据
        assert _ == md5_length
        # print(res, idx)  # 历史数据值
        # pdb.set_trace()

print(len(duplicate_res_list))

random.shuffle(duplicate_res_list)

jio.write_file_by_line(
    duplicate_res_list, '/home/cuichengyu/dataset/dupli_cws.txt')

# -----------------------------------------------------
'''  计算标签真实转移概率
dir_path = '/home/cuichengyu/github/jiojio/train_dir/temp'

trans_dict = {'B-B': 0, 'B-I': 0, 'I-B': 0, 'I-I': 0}
for text in jio.read_file_by_iter(os.path.join(dir_path, 'train.conll.txt')):
    assert type(text) is list
    tags = text[1]
    for tag_pre, tag_behind in zip(tags[:-1], tags[1:]):
        if tag_pre == 'B' and tag_behind == 'B':
            trans_dict['B-B'] += 1
        elif tag_pre == 'B' and tag_behind == 'I':
            trans_dict['B-I'] += 1
        elif tag_pre == 'I' and tag_behind == 'B':
            trans_dict['I-B'] += 1
        elif tag_pre == 'I' and tag_behind == 'I':
            trans_dict['I-I'] += 1
        else:
            print('the tag is wrong!')

#''
B-B: 0.34959443
B-I: 0.29970311
I-B: 0.29961667
I-I: 0.05108579
#''
total_num = sum(list(trans_dict.values()))
print(trans_dict)
# for key, value in trans_dict.items():
#    print('{}: {:.8f}'.format(key, value / total_num))
print('{}: {:.8f}'.format('B-B', trans_dict['B-B'] / (trans_dict['B-B'] + trans_dict['B-I'])))
print('{}: {:.8f}'.format('B-I', trans_dict['B-I'] / (trans_dict['B-B'] + trans_dict['B-I'])))

print('{}: {:.8f}'.format('I-B', trans_dict['I-B'] / (trans_dict['I-B'] + trans_dict['I-I'])))
print('{}: {:.8f}'.format('I-I', trans_dict['I-I'] / (trans_dict['I-B'] + trans_dict['I-I'])))
'''
