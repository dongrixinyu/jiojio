# **jiojio**
### - 简便中文分词器 a convenient Chinese word segmentation tool
<p align="center">
    <a alt="License">
        <img src="https://img.shields.io/github/license/dongrixinyu/jiojio?color=crimson" /></a>
    <a alt="Size">
        <img src="https://img.shields.io/badge/size-30.1m-orange" /></a>
    <a alt="Downloads">
        <img src="https://img.shields.io/pypi/dm/jiojio?color=yellow" /></a>
    <a alt="Version">
        <img src="https://img.shields.io/badge/version-1.1.4-green" /></a>
    <a href="https://github.com/dongrixinyu/jiojio/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/dongrixinyu/jiojio?color=blue" /></a>
</p>

# 适用场景
- 基于 **CPU** 的**高性能**、**持续优化** 中文分词器。

# 功能
- 基于 C 优化的 Python 接口分词器，单进程运行性能达 **5.2 万字/秒**，[**多个分词工具性能对比**](https://github.com/dongrixinyu/jiojio/wiki/多种常见开源分词工具的性能对比)
- 基于 CRF 算法，精细优化的 **字符特征工程**，[**模型特征说明**](https://github.com/dongrixinyu/jiojio/wiki/jiojio-分词CRF特征总结)
- 对模型文件的尽力压缩，**500万特征参数，模型文件大小30M**，方便 pip 安装
- 将词典加入模型，共同预测分词序列，流程一致性强，[**词典配置说明**](https://github.com/dongrixinyu/jiojio/wiki/向分词模型添加自定义词典)
- 将规则加入模型，有效克服某些类型文本受限于模型处理的情况，[**分词-添加正则**](../../wiki/jiojio-使用说明文档#user-content-分词-添加正则)
- 支持词性标注功能，与 [**JioNLP**](https://github.com/dongrixinyu/JioNLP) 联合实现**关键短语抽取**、**新闻地域识别** 等功能

# 安装
- pip 方式
```
$ pip install jiojio
```

- Git 方式
```
$ git clone https://github.com/dongrixinyu/jiojio
$ cd jiojio
$ pip install .
```

# 使用
- 基础方式
```
>>> import jiojio
>>> jiojio.init()
>>> print(jiojio.cut('我爱北京天安门！'))

# ['我', '爱', '北京', '天安门', '！']
# 可通过 jiojio.help() 获取基本使用方式说明
# 可通过 print(jiojio.init.__doc__) 获取模型初始化的各类参数
```

- 其它参数与设置
    - [**分词-添加正则**](../../wiki/jiojio-使用说明文档#user-content-分词-添加正则)
    - [**分词-增加词典**](../../wiki/jiojio-使用说明文档#user-content-分词-增加词典)
    - [**分词-调整 Viterbi 算子**](../../wiki/jiojio-使用说明文档#user-content-分词-调整-viterbi-算子)
    - [**词性标注**](../../wiki/jiojio-使用说明文档#user-content-词性标注)
    - [**词性标注-添加正则**](../../wiki/jiojio-使用说明文档#user-content-词性标注-添加正则)
    - [**词性标注-调整 Viterbi 算子**](../../wiki/jiojio-使用说明文档#user-content-词性标注-调整-viterbi-算子)

# 关于 jiojio 分词器的一些问答
- [与jiojio有关的问答](../../wiki/关于jiojio分词器的一些问答)

# TODO
- 开发分词在线版 [**JioNLP在线版**](http://182.92.160.94:16666/#/) 可快速试用分词功能
- 对分词器效果做**长期优化**
