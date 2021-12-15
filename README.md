# jiojio
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
- 对于某些 NLP 任务，需要依赖较高准确率的分词器与词性标注器，而当前主流的开源分词工具如 jieba、pkuseg 等则存在着准确率不高、维护终止等问题。
- 另一方面，NLP 领域的模型研究逐渐趋向于超大模型，强依赖 GPU 等设备；分词任务对于此类模型训练失去了意义，同时对于无 GPU 环境下高效的分词始终缺失。
- 因此，开发此款分词与词性标注工具，目的在于在 CPU 上实现高效分词，为依赖分词的 NLP 任务提供一种解决方案，常见解决方案见[JioNLP](https://github.com/dongrixinyu/JioNLP)。
