# jiojio
<p align="center">
    <a alt="License">
        <img src="https://img.shields.io/github/license/dongrixinyu/jiojio?color=crimson" /></a>
    <a alt="Size">
        <img src="https://img.shields.io/badge/size-30.1m-orange" /></a>
    <a alt="Downloads">
        <img src="https://img.shields.io/badge/downloads-10-yellow" /></a>
    <a alt="Version">
        <img src="https://img.shields.io/badge/version-0.0.1-green" /></a>
    <a href="https://github.com/dongrixinyu/jiojio/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/dongrixinyu/jiojio?color=blue" /></a>
</p>

a convenient Chinese word segmentation tool 简便中文分词器

# 适用场景
- 适用于试验、测试、基于 CPU 的上线服务、灵活的测试结果等等领域

# 功能
- 优化纯 Python 分词器，运行性能达 **2 万字/秒**
- 对字符特征有详细的优化过程，可进行特征增减，节约内存
- 将词典加入模型，共同预测分词序列，流程一致性强
- 支持多 CPU 多线程计算，线程启动可按需配置
- 在线版 [JioNLP在线版](http://182.92.160.94:16666/#/) 可快速试用分词功能
- 支持词性标注功能，与 [JioNLP](https://github.com/dongrixinyu/JioNLP) 联合实现**关键短语抽取**、**新闻地域识别** 等功能

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
>>> jiojio.init(dictionary='/path/to/dictionary.txt')
>>> words = jiojio.cut('欧盟各成员国内部也存在着种种分歧。')
>>> print(words)
# ['欧盟', '各', '成员国', '内部', '也, '存在', '着', '种种', '分歧', '。']
```

- 词典格式如下：
```
成员国\t1
比特币\t0.7
```


# TODO
- 对分词器做 C 加速优化
- 对分词器的自训练做优化
- 对分词器效果做**长期优化**

# 初衷
- 制作一个在 CPU 上的高速、优质分词工具。

#### 问：目前 NLP 开发与落地逐渐倾向于不分词直接用模型处理，分词的意义在哪里？
- **答**：很多 NLP 任务可以采用 Bert、GPT、Seq2Seq 来完成，但这不代表所有任务都可以依赖模型。鉴于当前模型对语义的理解仍停留在初级水平，依然需要词典、正则，需要大规模的泛化，那么分词就依然有用武之地。

#### 问：目前开源分词器如 jieba、pkuseg、lac、ansj 等，多如牛毛，为何还要再开发一个分词器？
- **答**：制作一个分词工具，在实际语料中达到 90%~94% 的 F1 值效果不难，但对于应用和落地，则需要 97%~99.9% 的 F1 值效果，这个优化过程需要详细的调试和分析，消耗大量的精力。
- 另一方面，NLP 领域的模型研究逐渐趋向于超大模型，强依赖 GPU 等设备；对于无 GPU 环境下高效的分词始终缺失。开发此款分词与词性标注工具，目的在于在 CPU 上实现高效分词。

#### 问：哪些任务依赖分词？
- **答**：很多定制化 NLP 任务依赖高速、优质的分词工具，常见解决方案见[JioNLP](https://github.com/dongrixinyu/JioNLP)。
