#!/bin/bash
# first you need to install Anaconda on Linux.

# make sure your python path is like the below ones.
# python: /home/ubuntu/anaconda3/bin/python
# pip: /home/ubuntu/anaconda3/bin/pip
# python3.6: /home/ubuntu/anaconda3/envs/py36/bin/python
# pip3.6: /home/ubuntu/anaconda3/envs/py36/bin/pip

# working in the directory with `setup.py` file.

source ~/.bashrc

# should designate jiojio version and python version
jiojio_version="1.2.1"
py3_min_version="6"
py3_max_version="10"

orig_python_path=`which python`
orig_pip_path=`which pip`
echo "python path: $orig_python_path"
echo "pip path: $orig_pip_path"
base_path=${orig_python_path%anaconda3*}
echo "base path: $base_path"

for (( i=$py3_min_version; i<=$py3_max_version; ++i ))
do
    echo "python version: 3.$i"
    conda create -n py3${i} python=3.${i} -y
    # conda activate py3${i}
    python_path="${base_path}anaconda3/envs/py3$i/bin/python"
    pip_path="${base_path}anaconda3/envs/py3$i/bin/pip"
    auditwheel_path="${base_path}anaconda3/envs/py3$i/bin/auditwheel"

    echo "python path: $python_path"
    echo "pip path: $pip_path"
    echo "auditwheel path: $auditwheel_path"

    # 删除 索引构建文件，避免影响下一次
    if [ -d build ]; then
        rm -rf build
    fi
    if [ -d jiojio.egg-info ]; then
        rm -rf jiojio.egg-info
    fi

    ${python_path} setup.py bdist_wheel --universal

    ${pip_path} install auditwheel
    ${auditwheel_path} repair ./dist/jiojio-${jiojio_version}*3${i}*whl
    # sleep 30s
    conda remove -n py3${i} --all -y

done

pip install twine
twine upload wheelhouse/jiojio-${jiojio_version}*whl

echo "finished!"
exit 0

# pip uninstall jiojio -y
# pip install -e .

# 以下代码是旧版本需每个py版本单独打包，可参考，无需考虑

# 对各个 Python 版本的打包，以 Python3.9 为例
# 在 jiojio 当前本文件所属目录下
conda create -n py39 python=3.9
conda activate py39

rm -rf build/ jiojio.egg-info/
python setup.py bdist_wheel --universal

conda deactivate
conda remove -n py39 --all -y

# 将 whl 转换为适合 pypi 接收的文件
pip install auditwheel
cd dist & auditwheel repair jiojio*39*whl

# 上传
pip install twine
twine upload wheelhouse/jiojio-${jiojio_version}*
