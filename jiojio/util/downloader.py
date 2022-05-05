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
import multiprocessing as mp

import zipfile
import requests


def unzip_file(zip_file_path):
    base_dir_path = os.path.dirname(zip_file_path)

    _zip_file = os.path.basename(zip_file_path)
    _zip_base_name = _zip_file.replace('.zip', '')

    target_dir_path = os.path.join(base_dir_path, _zip_base_name)
    if not os.path.exists(target_dir_path):
        os.mkdir(target_dir_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        # print(zf.namelist())
        for _file in zf.namelist():
            zf.extract(_file, target_dir_path)


def _download(_url, _zip_file_path, _file_name):
    try:
        import tqdm
    except:
        tqdm = None

    if tqdm is None:
        # 原始无 tqdm 版
        res = requests.get(_url)
        with open(_zip_file_path, 'wb') as fw:
            fw.write(res.content)

    else:
        response = requests.get(_url, stream=True)

        with tqdm.tqdm.wrapattr(
            open(_zip_file_path, 'wb'), 'write', miniters=1, desc=_file_name,
            total=int(response.headers.get('content-length', 0))) as fout:

            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)


def download_model(url, base_dir):
    """ 从远端下载模型压缩包 """
    file_name = url.split('/')[-1]
    print('Start downloading `{}` model. Please wait ...'.format(file_name))

    zip_file_path = os.path.join(base_dir, file_name)

    download_p = mp.Process(target=_download, args=(url, zip_file_path, file_name))
    download_p.start()
    download_p.join()

    unzip_file(zip_file_path)

    print('Successfully download `{}` model.'.format(file_name))
