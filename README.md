# **jiojio**
### - 基于CPU的高性能、持续迭代模型、简便中文分词器
### a convenient Chinese word segmentation tool
<p align="center">
    <a alt="License">
        <img src="https://img.shields.io/github/license/dongrixinyu/jiojio?color=crimson" /></a>
    <a alt="Size">
        <img src="https://img.shields.io/badge/size-82.1m-orange" /></a>
    <a alt="Downloads">
        <img src="https://pepy.tech/badge/jiojio/month" /></a>
    <a alt="Version">
        <img src="https://img.shields.io/badge/version-1.2.7-green" /></a>
    <a href="https://github.com/dongrixinyu/jiojio/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/dongrixinyu/jiojio?color=blue" /></a>
</p>

## 适用场景
- 基于 **CPU** 的**高性能**、**持续优化** 中文分词器。

## 功能
- 基于 C 的 Python 接口分词器，CPU 单进程运行性能达 **13.4 万字/秒**，[**多个分词工具性能对比**](https://github.com/dongrixinyu/jiojio/wiki/多种常见开源分词工具的性能对比)

- 网页版 [**JioNLP源站**](http://www.jionlp.com)，可快速试用分词、词性标注功能

- 基于 CRF 算法，精细优化的 **字符特征工程**，[**模型特征说明**](https://github.com/dongrixinyu/jiojio/wiki/jiojio-分词CRF特征总结)
- 对模型文件的尽力压缩，使用 `np.float8` 精度类型，**500万特征参数，模型文件大小30M**，方便 pip 安装
- 添加自定义词典兼容**静态、动态**两种方式，流程一致性强，[**词典配置说明**](https://github.com/dongrixinyu/jiojio/wiki/向分词模型添加自定义词典)
- 将规则加入模型，有效克服某些类型文本受限于模型处理的情况，[**分词-添加正则**](../../wiki/jiojio-使用说明文档#user-content-分词-添加正则)
- 支持词性标注功能，与 [**JioNLP**](https://github.com/dongrixinyu/JioNLP) 联合实现**关键短语抽取**、**新闻地域识别** 等功能

## 安装
- pip 方式（稳定版本）
```
$ pip install jiojio
```

- Git 方式（开发版本）
```
$ git clone https://github.com/dongrixinyu/jiojio
$ cd jiojio
$ pip install .
```

- 非 ubuntu 环境的 C 安装
如使用 windows 或 mac 等操作系统或其它硬件，则没有直接可调用 C 的库，程序默认直接调用纯 Python 进行分词，因此速度会慢。可以使用以下方式安装编译 C 库。以下方式仅供参考，在熟悉 C 语言后进行调试使用。[**关于windows平台的编译说明**](https://github.com/dongrixinyu/jiojio/wiki/%E5%85%B3%E4%BA%8E-Windows-%E5%B9%B3%E5%8F%B0%E7%BC%96%E8%AF%91%E7%9A%84%E8%AF%B4%E6%98%8E)
```
$ git clone https://github.com/dongrixinyu/jiojio
$ cd jiojio/jiojio/jiojio_cpp
$ ./compiler.sh
```

## 使用
- 基础方式
```
>>> import jiojio
>>> jiojio.init()
>>> print(jiojio.cut('开源软件应秉持全人类共享的精神，搞封闭式是行不通的。'))

# ['开源', '软件', '应', '秉持', '全人类', '共享', '的', '精神', '，', '搞', '封闭式', '是', '行', '不通', '的', '。']
# 可通过 jiojio.help() 获取基本使用方式说明
# 可通过 print(jiojio.init.__doc__) 获取模型初始化的各类参数
```

- 其它参数与设置
    - [**分词-添加正则**](../../wiki/jiojio-使用说明文档#user-content-分词-添加正则)
    - [**分词-增加静态词典**](../../wiki/jiojio-使用说明文档#user-content-分词-增加静态词典)
    - [**分词-增加动态词典**](../../wiki/jiojio-使用说明文档#user-content-分词-增加动态词典)
    - [**词性标注**](../../wiki/jiojio-使用说明文档#user-content-词性标注)
    - [**词性标注-添加正则**](../../wiki/jiojio-使用说明文档#user-content-词性标注-添加正则)
    - [**词性标注-增加动态词典**](../../wiki/jiojio-使用说明文档#user-content-词性标注-增加动态词典)

## 关于 jiojio 分词器的一些问答
- 可能早十年把这个分词器写出来，jiojio 也许现在就会流行起来。在 ChatGPT 称霸 NLP  界的今天，我写这个工具，加速这个工具，纯粹是为了提升一下 C 语言的编程能力。ChatGPT 能够做出来，还是需要理想主义的，我写这个工具同理。
- [与jiojio有关的问答](../../wiki/关于jiojio分词器的一些问答)

## TODO

- 对分词器效果做**标注数据更新**，模型**长期优化**

## 交流群聊

- 欢迎加入自然语言处理NLP交流群，搜索**wx公众号“JioNLP”**，或扫以下码即可入群

![image](https://github.com/dongrixinyu/JioNLP/blob/master/image/qrcode_for_gh.jpg)
