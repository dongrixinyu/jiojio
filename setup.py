# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'
# Website: http://www.jionlp.com


import os
import re
import sys
import setuptools


if sys.platform == 'linux':
    from setuptools import Extension
    from distutils.command.build_ext import build_ext as _build_ext


    class build_ext(_build_ext):

        def finalize_options(self):
            _build_ext.finalize_options(self)
            __builtins__.__NUMPY_SETUP__ = False

            import numpy
            self.include_dirs.append(numpy.get_include())
            self.libraries=[
                os.path.dirname(numpy.get_include()) + '/' + \
                [name for name in os.listdir(os.path.dirname(numpy.get_include()))
                 if '_multiarray_umath.' in name and name.endswith('.so')][0]]

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


# get version tag
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    readme_lines = f.readlines()
    version_pattern = re.compile(r'badge/version-(\d\.\d+\.\d+)-')
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
            Extension(
                'jiojio.jiojio_cpp.build.libcwsInterface',
                sources=['jiojio/jiojio_cpp/wchar_t_hash_set.c',
                 'jiojio/jiojio_cpp/wchar_t_hash_dict.c',
                 'jiojio/jiojio_cpp/cwsPrediction.c',
                 'jiojio/jiojio_cpp/cwsInterface.c'],
                # include_dirs=[np.get_include()],
                # # library_dirs=[os.path.dirname(np.get_include())],
                # libraries=[os.path.dirname(np.get_include()) + '/' + \
                #            [name for name in os.listdir(os.path.dirname(np.get_include()))
                #             if '_multiarray_umath.' in name and name.endswith('.so')][0]],
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
        cmdclass={'build_ext': build_ext} if sys.platform == 'linux' else {},
        setup_requires=['numpy'],
    )


if __name__ == '__main__':
    setup_package()
