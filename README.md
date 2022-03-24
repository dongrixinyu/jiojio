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
        <img src="https://img.shields.io/badge/version-1.1.1-green" /></a>
    <a href="https://github.com/dongrixinyu/jiojio/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/dongrixinyu/jiojio?color=blue" /></a>
</p>

# 适用场景
- 基于 **CPU** 的**高性能**、**持续优化**分词器。

# 功能
- 基于 C 优化的 Python 接口分词器，单进程运行性能达 **4.8 万字/秒**，[**多个分词工具性能对比**](https://github.com/dongrixinyu/jiojio/wiki/多种常见开源分词工具的性能对比)
- 基于 CRF 算法，精细优化的 **字符特征选择**，可进行特征增减，节约内存，[**模型特征说明**](https://github.com/dongrixinyu/jiojio/wiki/jiojio-分词CRF特征总结)
- 将词典加入模型，共同预测分词序列，流程一致性强，[**词典配置说明**](https://github.com/dongrixinyu/jiojio/wiki/向分词模型添加自定义词典)
- 支持多 CPU 多线程计算，线程启动可按需配置
- 在线版 [**JioNLP在线版**](http://182.92.160.94:16666/#/) 可快速试用分词功能
- 支持词性标注功能，与 [**JioNLP**](https://github.com/dongrixinyu/JioNLP) 联合实现**关键短语抽取**、**新闻地域识别** 等功能

# 安装
- Git 方式
```
$ git clone https://github.com/dongrixinyu/jiojio
$ cd jiojio
$ pip install .
```

- pip 方式
```
$ pip install jiojio
```

# 使用
```
>>> import jiojio
>>> jiojio.init()
>>> words = jiojio.cut('我爱北京天安门！')
>>> print(words)

# ['我', '爱', '北京', '天安门', '！']

```

- 增加词典
```
>>> import jiojio
>>> jiojio.init(user_dict='/path/to/dictionary.txt')
>>> words = jiojio.cut('欧盟各成员国内部也存在着种种分歧。')
>>> print(words)
# ['欧盟', '各', '成员国', '内部', '也, '存在', '着', '种种', '分歧', '。']
```

- 词典格式参考 [**用户自定义词典**](https://github.com/dongrixinyu/jiojio/blob/master/user_dict.txt)


# TODO
- 对分词器的自训练做优化
- 对分词器效果做**长期优化**

# 初衷
- 制作一个在 CPU 上的高速、优质分词工具。

- **问：目前 NLP 开发与落地逐渐倾向于不分词直接用模型处理字 token，分词的意义在哪里？**
- **答**：很多 NLP 任务可以采用 Bert、GPT、Seq2Seq 来完成，但这不代表所有任务都可以依赖模型。很多**信息抽取、解析依然需要基于高效的分词来处理**，需要词典、正则，需要大规模的泛化，那么高效的分词工具就依然有用武之地。

- **问：目前开源分词器如 jieba、pkuseg、lac、ansj 等，多如牛毛，为何还要再开发一个分词器？**
- **答**：制作一个分词工具，在实际语料中达到 90%~94% 的 F1 值效果不难，但对于应用和落地，则需要 95%~99.9% 的 F1 值效果，这个优化过程需要详细的调试和分析，消耗大量的精力。
- 另一方面，NLP 领域的模型研究逐渐趋向于超大模型，强依赖 GPU 等设备；对于**无 GPU 环境下高效的分词始终缺失**。开发此款分词与词性标注工具，目的在于在 CPU 上实现高效分词。

- **问：语言随社会变化发展演变很快，每年都有新兴词汇出现，如何保证模型的时效性和准确性？**
- **答**：分词工具的模型优化主要核心点在于新语料数据的更新。因此，本工具提供了界面化的新语料提交接口。用户可以自主向云端提供新语料，模型计算错误的语料，本工具会定期进行模型的优化和重新训练。

- **问：哪些任务依赖高效的分词？**
- **答**：很多定制化 NLP 任务依赖高速、优质的分词工具，常见解决方案见[JioNLP](https://github.com/dongrixinyu/JioNLP)。
