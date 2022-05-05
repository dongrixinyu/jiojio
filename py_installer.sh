pip uninstall jiojio -y
pip install -e .

# 对各个 Python 版本的打包，以 Python3.9 为例
# 在 jiojio 当前本文件所属目录下
conda create -n py39 python=3.9
conda activate py39

rm -rf build/ jiojio.egg-info/
python setup.py bdist_wheel --universal

conda deactivate
conda remove -n py39 --all

# 将 whl 转换为适合 pypi 接收的文件
pip install auditwheel
cd dist & auditwheel repair jiojio*39*whl

# 上传
pip install twine
twine upload wheelhouse/*
