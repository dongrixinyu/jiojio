import os
import numpy as np

import jionlp as jio

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

'''
B-B: 0.34959443
B-I: 0.29970311
I-B: 0.29961667
I-I: 0.05108579
'''
total_num = sum(list(trans_dict.values()))
print(trans_dict)
# for key, value in trans_dict.items():
#    print('{}: {:.8f}'.format(key, value / total_num))
print('{}: {:.8f}'.format('B-B', trans_dict['B-B'] / (trans_dict['B-B'] + trans_dict['B-I'])))
print('{}: {:.8f}'.format('B-I', trans_dict['B-I'] / (trans_dict['B-B'] + trans_dict['B-I'])))

print('{}: {:.8f}'.format('I-B', trans_dict['I-B'] / (trans_dict['I-B'] + trans_dict['I-I'])))
print('{}: {:.8f}'.format('I-I', trans_dict['I-I'] / (trans_dict['I-B'] + trans_dict['I-I'])))
