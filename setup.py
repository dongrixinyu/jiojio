# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'


import os
import re
import sys
import setuptools


if sys.platform == 'linux':
    from setuptools import Extension
    from distutils.command.build_ext import build_ext


    class build_ext(build_ext):

        def build_extension(self, ext):
            self._ctypes = isinstance(ext, CTypes)
            return super().build_extension(ext)

        def get_export_symbols(self, ext):
            if self._ctypes:
                return ext.export_symbols
            return super().get_export_symbols(ext)

        def get_ext_filename(self, ext_name):
            if self._ctypes:
                return ext_name + '.so'
            return super().get_ext_filename(ext_name)


    class CTypes(Extension):
        pass


DIR_PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    readme_lines = f.readlines()
    version_pattern = re.compile('badge/version-(\d\.\d+\.\d+)-')
    for line in readme_lines:
        result = version_pattern.search(line)
        if result is not None:
            __version__ = result.group(1)

    LONG_DOC = '\n'.join(readme_lines)


with open(os.path.join(DIR_PATH, 'requirements.txt'),
          'r', encoding='utf-8') as f:
    requirements = f.readlines()


def setup_package():
    if sys.platform == 'linux':
        extensions = [
            Extension(
                'jiojio.jiojio_cpp.build.libcwsFeatureToIndex',
                ['jiojio/jiojio_cpp/cwsFeatureToIndex.c'],
                language='c'),
            Extension(
                'jiojio.jiojio_cpp.build.libcwsFeatureExtractor',
                ['jiojio/jiojio_cpp/cwsFeatureExtractor.c'],
                language='c'),
            Extension(
                'jiojio.jiojio_cpp.build.libtagWordsConverter',
                ['jiojio/jiojio_cpp/tagWordsConverter.c'],
                language='c'),
            Extension(
                'jiojio.jiojio_cpp.build.libposFeatureExtractor',
                ['jiojio/jiojio_cpp/posFeatureExtractor.c'],
                language='c'),
        ]
    else:
        extensions = None

    # 本行参数用于构建纯 py wheel 文件
    # extensions = None
    setuptools.setup(
        name='jiojio',
        version=__version__,
        author='dongrixinyu',
        author_email='dongrixinyu.89@163.com',
        description='jiojio: a convenient Chinese word segmentation tool',
        long_description=LONG_DOC,
        long_description_content_type='text/markdown',
        url='https://github.com/dongrixinyu/jiojio',
        packages=setuptools.find_packages(),
        package_data={
            '': ['*.txt', '*.pkl', '*.npz', '*.zip',
                 '*.json', '*.c', '*.h']
        },
        include_package_data=True,
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Topic :: Text Processing',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        install_requires=requirements,
        ext_modules=extensions,
        zip_safe=False,
        cmdclass={'build_ext': build_ext} if sys.platform == 'linux' else {}
    )


if __name__ == '__main__':
    setup_package()
