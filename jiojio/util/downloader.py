# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'


import os
import pdb
import sys


def unzip_file(zip_file_path):
    base_dir_path = os.path.dirname(zip_file_path)

    _zip_file = os.path.basename(zip_file_path)
    _zip_base_name = _zip_file.replace('.zip', '')

    target_dir_path = os.path.join(base_dir_path, _zip_base_name)
    if os.path.exists(target_dir_path):
        os.mkdirs(target_dir_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        # print(zf.namelist())
        for _file in zf.namelist():
            zf.extract(_file, target_dir_path)


def download_model(url, base_dir):
    """ 从远端下载模型压缩包 """
    file_name = url.split('/')[-1]
    res = requests.get(url)

    zip_file_path = os.path.join(base_dir, file_name)
    with open(zip_file_path, 'wb') as fw:
        fw.write(res.content)

    unzip_file(zip_file_path)

    print('Successfully download `{}` model.'.format(file_name))
