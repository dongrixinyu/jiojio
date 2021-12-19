import setuptools
import os
# from distutils.extension import Extension
import re
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR_PATH, 'README.md'), 'r', encoding='utf-8') as f:
    readme_lines = f.readlines()
    version_pattern = re.compile('badge/version-(\d\.\d+\.\d+)-')
    for line in readme_lines:
        result = version_pattern.search(line)
        if result is not None:
            __version__ = result.group(1)

    LONG_DOC = '\n'.join(readme_lines)


def setup_package():
    '''
    extensions = [
        Extension(
            "jiojio.inference",
            ["jiojio/inference.pyx"],
            include_dirs=[np.get_include()],
            language="c++"
        ),
        Extension(
            "jiojio.feature_extractor",
            ["jiojio/feature_extractor.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "jiojio.postag.feature_extractor",
            ["jiojio/postag/feature_extractor.pyx"],
            include_dirs=[np.get_include()],
        ),
    ]

    def is_source_release(path):
        return os.path.exists(os.path.join(path, "PKG-INFO"))

    if not is_source_release(DIR_PATH):
        from Cython.Build import cythonize
        extensions = cythonize(extensions, annotate=True)
    '''
    setuptools.setup(
        name="jiojio",
        version="0.0.1",
        author="dongrixinyu",
        author_email="dongrixinyu.89@163.com",
        description="jiojio: a convenient Chinese word segmentation tool",
        long_description=LONG_DOC,
        long_description_content_type="text/markdown",
        url="https://github.com/dongrixinyu/jiojio",
        packages=setuptools.find_packages(),
        package_data={
            "": ["*.txt*", "*.pkl", "*.npz", "*.pyx", "*.pxd", "*.zip"]
        },
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: OS Independent",
        ],
        install_requires=["numpy>=1.16.0"],
        # setup_requires=["cython", "numpy>=1.16.0"],
        # ext_modules=extensions,
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
