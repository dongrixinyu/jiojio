# -*- coding=utf-8 -*-

import os
import pdb
import shutil
import zipfile


FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(os.path.dirname(FILE_PATH))


def zip_file(file_path):
    """ 将某些 txt, json 文件压缩，须指定绝对路径 """
    dir_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    # 必须复制一下
    tmp_file_path = os.path.join(os.getcwd(), file_name)
    shutil.copyfile(file_path, tmp_file_path)

    zip_file_name = file_name.split('.')[0] + '.zip'

    with zipfile.ZipFile(os.path.join(dir_path, zip_file_name),
            'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_name)

    os.remove(tmp_file_path)


def unzip_file(zip_file_path):
    """ 将某些 txt 文件解压缩，须指定绝对路径 """
    zip_dir_path = os.path.dirname(zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        assert len(zf.namelist()) == 1
        for _file in zf.namelist():
            zf.extract(_file, zip_dir_path)
