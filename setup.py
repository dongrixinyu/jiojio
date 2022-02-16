# -*- coding=utf-8 -*-

import os
import re
import setuptools

from setuptools import Extension
from distutils.command.build_ext import build_ext


DIR_PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    readme_lines = f.readlines()
    version_pattern = re.compile('badge/version-(\d\.\d+\.\d+)-')
    for line in readme_lines:
        result = version_pattern.search(line)
        if result is not None:
            __version__ = result.group(1)

    LONG_DOC = '\n'.join(readme_lines)


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


class CTypes(Extension): pass


def setup_package():

    extensions = [
        Extension(
            'jiojio.jiojio_cpp.build.libfeatureExtractor',
            ['jiojio/jiojio_cpp/featureExtractor.c'],
            language='c'
        ),
        Extension(
            'jiojio.jiojio_cpp.build.libtagWordsConverter',
            ['jiojio/jiojio_cpp/tagWordsConverter.c'],
            language='c'
        ),
    ]

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
            'License :: Other/Proprietary License',
            'Operating System :: OS Independent',
        ],
        install_requires=['numpy>=1.16.0'],
        ext_modules=extensions,
        zip_safe=False,
        cmdclass={'build_ext': build_ext}
    )


if __name__ == '__main__':
    setup_package()
