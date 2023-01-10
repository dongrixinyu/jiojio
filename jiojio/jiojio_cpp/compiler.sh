#!/bin/bash

if [ ! -d build ]; then
    mkdir build
else
    rm -rf build/*
fi

cd build
cmake .. \
    -DPYTHON_INCLUDE_DIRS=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARIES=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))")  \
    -DNUMPY_INCLUDE_DIRS=$(python -c "import numpy as np; print(np.get_include())")  \
    -DNUMPY_LIBRARIES=$(python -c "import numpy as np; print(np.core._multiarray_umath.__file__)")
make

# cd .. && python test_cpp.py
