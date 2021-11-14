## 去除停用词

#### remove_stopwords

给定一串经过分词后的词汇组成的 list，去除其中的停用词。

```
>>> import pkuseg
>>> import jionlp as jio
>>> text = '2018年发行的《新华词典》，是一部以语文为主兼收百科的中型词典，适合中学师生及中等以上文化程度的读者使用。'
>>> pkuseg_obj = pkuseg.pkuseg()
>>> text_segs = pkuseg_obj.cut(text)
>>> res1 = jio.remove_stopwords(text_segs)
>>> res2 = jio.remove_stopwords(text_segs, remove_time=True)
>>> print(res1)
>>> print(res2)

# text_segs: ['2018年', '发行', '的', '《', '新华', '词典', '》', '，', '是', '一', '部', '以', '语文', 
#             '为主', '兼收百科', '的', '中型', '词典', '，', '适合', '中学', '师生', '及', '中等', '以上', 
#             '文化', '程度', '的', '读者', '使用', '。']
# res1: ['2018年', '发行', '新华', '词典', '一', '部', '语文', '兼收百科', '中型', '词典', '适合', '中学',
#        '师生', '中等', '文化', '程度', '读者', '使用']
# res2: ['发行', '新华', '词典', '一', '部', '语文', '兼收百科', '中型', '词典', '适合', '中学', '师生', 
#        '中等', '文化', '程度', '读者', '使用']

```

- 若须调整词典，需要进入工具包的 ```jionlp/dictionary/stopwords.txt``` 文件直接修改。
- 工具提供了删除**时间**词汇的参数```remove_time(bool)```，可以将词汇列表中的**年月日、季节、早中晚**等词汇剔除。如上例所示。具体正则表达式可进入工具包的```jionlp/rule/rule_pattern.py``` 文件查看修改。
    - 文本中，时间词汇分为两种语言功能，一种是作为名词性成分，另一种是作为时间状语成分。
    - 本工具依据此划分，保留名词性成分的模糊时间词汇，如“三十多年”、“六七个月”等。
    - 删除时间状语成分，如“2019年3月10日”、“第一季度”、“18:30:51”、“3~4月份”、“清晨”、“年前”等。
    - 该区分方法较为笼统，但核心目的是去除具体指示时间，保留虚指模糊时间。更详细信息在```jionlp/gadget/remove_stopwords.py```。
- 工具提供了删除**地名**词汇的参数```remove_location```，可以将词汇列表中的具体地名等词汇删除，如“宁夏”、“英国”、“沙溪镇”、“珊瑚海”、“艾斯卡丁郡”等。
- 工具提供了删除**纯数字**词汇的参数```remove_number```，可以将词汇列表中的纯数字等词汇删除，如“12900”、“十万三千多”、“百分之六十七”、“0.0123”等。
- 工具提供了删除**非中文字符**词汇的参数```remove_non_chinese```，可以将词汇列表中的非中文字符等词汇删除，如“-----”、“###”、“abs~”等。
- 工具提供了保留**否定**词汇的参数```save_negative_words```，可以将词汇列表中的否定词汇保留，如“没有”、“不”、“非”等。


----------

## 文本分句

#### split_sentence

给定一段文本，按照中文标点符号做**分句**。
```
>>> import jionlp as jio
>>> text = '他说：“中华古汉语，泱泱大国，历史传承的瑰宝。。。”'
>>> res = jio.split_sentence(text, criterion='fine')
>>> print(res)

# ['他说：', '“中华古汉语，', '泱泱大国，', '历史传承的瑰宝。。。”']
```

- 参数 criterion 分为 ```coarse```粗粒度 和 ```fine```细粒度 两种；
    - 粗粒度按照```。！？“”```等中文完整句子语义来确定；
    - 细粒度按照```。！？，：；、“”‘’``` 等中文短句来确定。
- 引号的处理依照与前后文本结合紧密度来确定，如上例所示。

-----------

## 地址解析

#### parse_location

给定一个（地址）字符串，识别其中的**省、市、县三级地名**，指定参数```town_village(bool)```，可获取**乡镇、村、社区两级详细地名**，指定参数```change2new(bool)```可自动将旧地址转换为新地址。

```
# 例 1
>>> import jionlp as jio
>>> text = '武侯区红牌楼街道19号红星大厦9楼2号'
>>> res = jio.parse_location(text, town_village=True)
>>> print(res)

# {'province': '四川省',
#  'city': '成都市',
#  'county': '武侯区',
#  'town': '红牌楼街道',
#  'village': None,
#  'detail': '红牌楼街道19号红星大厦9楼2号',
#  'full_location': '四川省成都市武侯区红牌楼街19号红星大厦9楼2号',
#  'orig_location': '武侯区红牌楼街19号红星大厦9楼2号'}

# 例 2
>>> text = '四川金阳2019年易地扶贫搬迁工程'
>>> res = jio.parse_location(text)
>>> print(res)

# {'province': '四川省', 
#  'city': '凉山彝族自治州', 
#  'county': '金阳县', 
#  'detail': '2019年易地扶贫搬迁工程', 
#  'full_location': '四川省凉山彝族自治州金阳县2019年易地扶贫搬迁工程', 
#  'orig_location': '四川金阳2019年易地扶贫搬迁工程'}

# 例 3：自动将旧地名 港闸 映射至新地名 崇川，2020年国务院批准
>>> text = '港闸区陈桥街道33号'
>>> res = jio.parse_location(text, change2new=True, town_village=True)  
>>> print(res)

# {'province': '江苏省', 
#  'city': '南通市', 
#  'county': '崇川区', 
#  'detail': '陈桥街道33号', 
#  'full_location': '江苏省南通市崇川区陈桥街道33号',
#  'orig_location': '港闸区陈桥街道33号'}

```

- 若字符串中缺少省市信息，可依据词典做自动补全，如上例1中，根据“武侯区” 补全 “四川、成都”。
- 若字符串中仅有 “高新区”，无法做补全，则按原样返回结果。
- 字符串不局限于地址，如上例2，若不包含任何地址，则返回为最靠前出现的地址。
- 若地址名仅仅为简称，如上例2，在不存在歧义的前提下，会自动补全。
- 词典位置：```jionlp/dictionary/china_location.txt```，该词典从 [2020年国家统计局行政区划](http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020)得到
- 若地址中涉及多个省市县，则以最靠前的地址为准，仅匹配一个。
- 该词典可返回省、市、县三级，再指定参数```town_village(bool)```，可获取**乡镇、村、社区两级详细地名**，但是乡镇、村社两级地址必须使用全名匹配，不支持简称；同时必须在省、市、县指定完全清晰的情况下才生效。
- 国内行政区划有变动，支持使用```change2new(bool)```自动将旧地名转换为新地名，仅限省、市、县三级的转换，如上例3。若该选项为```False```，则```town_village(bool)```无法抽取旧地址中的乡镇与街道。

-----------

## 新闻地名识别

#### recognize_location

给定一篇新闻，识别其中的**国内省、市、县地名，国外国家、城市名**，并以层级结构返回。该方法多用于舆情统计分析。

```
>>> import jionlp as jio
>>> text = '海洋一号D星。中新网北京6月11日电(郭超凯)记者从中国国家航天局获悉，6月11日2时31分，在牛家村，中国在太原卫星发射中心用长征二号丙运载火箭成功发射海洋一号D星。该星将与海洋一号C星组成中国首个海洋民用业务卫星星座。相比于美国，海洋一号D星是中国第四颗海洋水色系列卫星，是国家民用空间基础设施规划的首批海洋业务卫星之一。'
>>> res = jio.recognize_location(text)
>>> print(res)

# {
#     "domestic":[
#         [{"province":"北京市", "city":"北京市", "county":null}, 2],
#         [{"province":"山西省", "city":"太原市", "county":null}, 1]
#     ],
#     "foreign":[
#         [{"country":"中国", "city":"北京"}, 5],
#         [{"country":"美国", "city":null}, 1]
#     ],
#     "others":{"牛家村":1}
# }

```

- 采用了北大分词器 pkuseg，词性为 ns 地名的词汇进行统计，计算效果和性能 80% 程度上受到分词器影响；
- 当有多个地址返回时，排序靠后的地址往往可靠性低；
- 国内地名未考虑乡镇级，国外地名未考虑洲、州、邦、县级；地名未考虑海、河、山、楼等，此类全部存入 others 字段；
- 文本中，存在“中国”二字，往往为外交新闻，也可能出现在 ```foreign``` 字段中，如上例；
- 词典位置：```jionlp/dictionary/china_location.txt```，该词典从 [2020年中国行政区划](http://www.mca.gov.cn/article/sj/xzqh/2020/2020/202003061536.html)，以及```jionlp/dictionary/world_location.txt```
- TODO：如“北京时间”不能计入地名计算，往往分词器无法判断；“日美同盟”中需要分别考虑日本、美国；
- 返回结果中，第一个地址属于文本的归属地的正确率为 93%。

--------------

## 身份证号码解析

#### parse_id_card

给定一个身份证号码，解析其对应的**省、市、县、出生年月、性别、校验码**

```
>>> text = '52010320171109002X'
>>> res = jio.parse_id_card(text)
>>> print(res)

# {'province': '贵州省',
#  'city': '贵阳市',
#  'county': '云岩区',
#  'birth_year': '2017',
#  'birth_month': '11',
#  'birth_day': '09',
#  'gender': '女',
#  'check_code': 'x'}

```

- 若给定字符串不是身份证号，返回为 None
- 某些行政区划码已被撤销，如 140402（原山西省长治市城区），但仍有此类身份证号，此时仅能解析部分（山西省长治市）


----------

## 繁体转简体字

#### tra2sim

给定一段文本，将其中的繁体字转换为简体字，提供```char``` 和 ```word```两种模式，区别如下：
```
>>> import jionlp as jio
>>> text = '今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？'
>>> res1 = jio.tra2sim(text, mode='char')
>>> res2 = jio.tra2sim(text, mode='word')
>>> print(res1)
>>> print(res2)

# 今天天气好晴朗，想吃速食面。你还在工作吗？在太空梭上工作吗？
# 今天天气好晴朗，想吃方便面。你还在工作吗？在航天飞机上工作吗？
```

- ```char``` 模式是按照字符逐个替换为简体字
- ```word``` 模式是将港台地区的词汇表述习惯，替换为符合大陆表述习惯的相应词汇
- 采用前向最大匹配的方式执行

----------

## 简体转繁体字

#### sim2tra

给定一段文本，将其中的简体字转换为繁体字，提供```char``` 和 ```word```两种模式，区别如下：
```
>>> import jionlp as jio
>>> text = '今天天气好晴朗，想吃方便面。你还在工作吗？在航天飞机上工作吗？'
>>> res1 = jio.sim2tra(text, mode='char')
>>> res2 = jio.sim2tra(text, mode='word')
>>> print(res1)
>>> print(res2)

# 今天天氣好晴朗，想喫方便面。妳還在工作嗎？在航天飛機上工作嗎？
# 今天天氣好晴朗，想喫速食麵。妳還在工作嗎？在太空梭上工作嗎？
```

- ```char``` 模式是按照字符逐个替换为繁体字
- ```word``` 模式是将大陆的词汇表述习惯，替换为符合港台表述习惯的相应词汇
- 采用前向最大匹配的方式执行


----------

## 汉字转拼音

#### pinyin

给定一段文本，将其中的汉字标注汉语拼音，提供```standard(zhòng)```、```simple(zhong4)```和```detail(声母、韵母、声调)```三种模式：
```
>>> import jionlp as jio
>>> text = '中华人民共和国。'
>>> res1 = jio.pinyin(text)
>>> res2 = jio.pinyin(text, formater='simple')
>>> res3 = jio.pinyin('中国', formater='detail')
>>> print(res1)
>>> print(res2)
>>> print(res3)

# ['zhōng', 'huá', 'rén', 'mín', 'gòng', 'hé', 'guó', '<py_unk>']
# ['zhong1', 'hua2', 'ren2', 'min2', 'gong4', 'he2', 'guo2', '<py_unk>']
# [{'consonant': 'zh', 'vowel': 'ong', 'tone': '1'}, 
#  {'consonant': 'g', 'vowel': 'uo', 'tone': '2'}]
```

- 对于非汉字字符，以及非常用汉字字符（如仅用于韩文和日文的汉字字符），该工具直接返回```<py_unk>```
- ```standard``` 模式返回标准的汉语拼音。
- ```simple``` 模式将字母和注音分离，更适合用于深度学习模型建模。
- ```detail``` 模式返回声母（consonant）、韵母（vowel）、声调（tone）信息。其中声母共计23个，韵母共计34个，声调共计5个，轻声以数字```5```标记。
- 采用正向最大匹配，优先匹配多音词汇和短语。

----------

## 汉字转偏旁与字形

#### char_radical

给定一段文本，将其中的汉字标注**偏旁部首**与**字形结构**。字形结构分为 9 种，使用```jio.STRUCTURE_DICT```查看。
同时给出**四角编码**、**拆字部件**和**五笔编码**信息。
```
>>> import jionlp as jio
>>> text = '植树节是哪一天呢？'
>>> res = jio.char_radical(text)
>>> print(jio.STRUCTURE_DICT)
>>> print(res)

# {0: '一体结构', 1: '左右结构', 2: '上下结构', 3: '左中右结构', 4: '上中下结构', 
#  5: '右上包围结构', 6: '左上包围结构', 7: '左下包围结构', 8: '全包围结构', 9: '半包围结构'}
# [{'radical': '木', 'structure': '左右结构', 'corner_coding': '44912', 'stroke_order': '木直', 'wubi_coding': 'SFHG'}
#  {'radical': '木', 'structure': '左中右结构', 'corner_coding': '44900', 'stroke_order': '木又寸', 'wubi_coding': 'SCFY'}
#  {'radical': '草', 'structure': '上下结构', 'corner_coding': '44227', 'stroke_order': '竹卩', 'wubi_coding': 'ABJ'}
#  {'radical': '日', 'structure': '上下结构', 'corner_coding': '60801', 'stroke_order': '日疋', 'wubi_coding': 'JGHU'}
#  {'radical': '口', 'structure': '左右结构', 'corner_coding': '67027', 'stroke_order': '口那', 'wubi_coding': 'KVFB'}
#  {'radical': '一', 'structure': '一体结构', 'corner_coding': '10000', 'stroke_order': '一', 'wubi_coding': 'GGLL'}
#  {'radical': '大', 'structure': '上下结构', 'corner_coding': '10804', 'stroke_order': '一大', 'wubi_coding': 'GDI'}
#  {'radical': '口', 'structure': '左右结构', 'corner_coding': '67012', 'stroke_order': '口尼', 'wubi_coding': 'KNXN'}
#  {'radical': '<cr_unk>', 'structure': '一体结构', 'corner_coding': '00000', 'stroke_order': '<so_unk>', 'wubi_coding': 'XXXX'}]

```

- 对于非汉字字符，以及非常用汉字字符（如仅用于韩文和日文的汉字字符），该工具直接返回偏旁未定义```<cr_unk>```，字体结构```一体结构```，四角编码```00000```，笔画顺序```<so_unk>```，五笔编码```XXXX```。
- 一些汉字有多个偏旁部首，如“岡”，既包括“山”，也包括“冂”，其字本意为“山脊”，因此在指定偏旁时，指定为“山”。
- 一些变形偏旁，如“艹”、“氵”等，直接使用其原意汉字替代，如“草”、“水”等。方便直接使用对应汉字的 embedding
- 四角编码信息是基于笔画、位置信息构造的，与部首、结构信息有重复冗余之处。
- 拆字部件未转化为标准汉字，仍以偏旁形式存在。
- 五笔编码在一定程度上体现了汉字的结构。


----------

## 关键短语抽取

#### extract_keyphrase

给定一段文本，返回其中的关键短语，默认为5个。

```
>>> import jionlp as jio
>>> text = '朝鲜确认金正恩出访俄罗斯 将与普京举行会谈...'
>>> key_phrases = jio.keyphrase.extract_keyphrase(text)
>>> print(key_phrases)
>>> print(jio.keyphrase.extract_keyphrase.__doc__)

# ['俄罗斯克里姆林宫', '邀请金正恩访俄', '举行会谈',
#  '朝方转交普京', '最高司令官金正恩']

```

- 原理简述：在 tfidf 方法提取的碎片化的关键词（默认使用 pkuseg 的分词工具）基础上，将在文本中相邻的关键词合并，并根据权重进行调整，同时合并较为相似的短语，并结合 LDA 模型，寻找突出主题的词汇，增加权重，组合成结果进行返回。
- 参数较多，可调节部分也较灵活，可以参考 ```print(jio.keyphrase.extract_keyphrase.__doc__)```
- 更细节的用法可以参考 [CKPE](https://github.com/dongrixinyu/chinese_keyphrase_extractor)
- ```pos_combine_weight.json```与```topic_word_weight.json```文件的计算方法参考[各文件的计算方法](https://github.com/dongrixinyu/chinese_keyphrase_extractor/wiki/%E5%90%84%E4%B8%AA%E7%BB%9F%E8%AE%A1%E6%96%87%E4%BB%B6%E7%9A%84%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95)

- 更新目标：1、同义词合并提升短语权重；2、抽取关键动词，将其纳入关键短语

### 扩展应用一：扩展类型短语

- 有时产品和客户给定了一些词汇列表，比如化工经营业务词汇“聚氯乙烯”、“塑料”、“切割”、“金刚石”等。想要找到跟这些词汇相关的短语。
```
import jionlp as jio

text = '聚氯乙烯树脂、塑料制品、切割工具、人造革、人造金刚石、农药（不含危险化学品）、针纺织品自产自销。...'
word_dict = {'聚氯乙烯': 1, '塑料': 1, '切割': 1, '金刚石': 1}  # 词汇: 词频（词频若未知可全设 1）
key_phrases = jio.keyphrase.extract_keyphrase(text, top_k=-1, specified_words=word_dict)
print(key_phrases)
```

### 扩展应用二：扩充NER特定类型实体

- 在做NER命名实体识别任务的时候，我们需要从文本中，将已有的类型词汇做扩充，如“机构”类别，但我们仅知道机构的一些特征，如常以“局”、“法院”、“办公室”等特征词结尾。
```
import jionlp as jio

text = '国务院下发通知，山西省法院、陕西省检察院、四川省法院、成都市教育局。...'
word_dict = {'局': 1, '国务院': 1, '检察院': 1, '法院': 1}
key_phrases = jio.keyphrase.extract_keyphrase(text, top_k=-1, specified_words=word_dict, 
                                         remove_phrases_list=['麻将局'])
print(key_phrases)
```
----------

## 成语接龙

#### idiom_solitaire

给定一条成语，返回其尾字为首的成语。

```
idiom = input('input: ')
n = 0
while n < 10:
    idiom = jio.idiom_solitaire(idiom, same_pinyin=False, same_tone=True)
    print('A: ', idiom)
    idiom = jio.idiom_solitaire(idiom, same_pinyin=False, same_tone=True)
    print('B: ', idiom)
    n += 1

# 执行后，工具代码会以 A 和 B 两个角色无限把成语接龙玩下去
```

- cur_idiom(str): 当前输入的成语，为其寻找下一个接龙成语
- check_idiom(bool): 检查当前输入的 cur_idiom 是否是成语，默认为 False
- same_pinyin(bool): 拼音一致即可接龙，否则必须同一个汉字才可接龙，默认 True
- same_tone(bool): same_pinyin 为 True 时有效，即拼音的音调一致才可接龙，否则算错，默认为 True
- with_prob(bool): 以成语的使用频率进行返回，即常见成语更容易返回，否则更易返回罕见成语
- restart(bool): 重新开始新一轮成语接龙，即清空已使用成语列表，默认 False

----------

## 抽取式文本摘要

#### extract_summary

给定一段文本，返回其抽取式的文本摘要，默认200字以内。

```
>>> import jionlp as jio
>>> text = '海外网11月10日电当地时间9日，美国总统特朗普在推特上发文表示，美国国防部长马克·埃斯珀已经被开除。...'
>>> res = jio.summary.extract_summary(text)
>>> print(res)

# 特朗普的推文写道：“马克 埃斯珀已经被开除。...
```

- 原理简述：为每个文本中的句子分配权重，权重计算包括 tfidf 方法的权重，以及 LDA 主题权重，以及 lead-3 得到位置权重，同时将长度低于15，大于70的句子权重做削减。并在最后结合 MMR 模型对句子做筛选，得到抽取式摘要。（默认使用 pkuseg 的分词工具效果好）
- 参数较多，可调节部分也较灵活，可以参考 ```print(jio.summary.extract_summary.__doc__)```
- 本工具仍有很大提升空间，此处作为 baseline 。

- ```pos_combine_weight.json```与```topic_word_weight.json```文件的计算方法参考[各文件的计算方法](https://github.com/dongrixinyu/chinese_keyphrase_extractor/wiki/%E5%90%84%E4%B8%AA%E7%BB%9F%E8%AE%A1%E6%96%87%E4%BB%B6%E7%9A%84%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95)
-----------------------------

## 电话号码归属地、运营商解析

#### jio.phone_location

给定一个电话号码字符串，识别其中的**省、市二级地名**，**手机运营商**。

```
>>> import jionlp as jio
>>> text = '联系电话：13288568202. (021)32830431'
>>> num_list = jio.extract_phone_number(text)
>>> print(num_list)
>>> res = [jio.phone_location(item['text']) for item in num_list]
>>> print(res)

# [{'text': '13288568202', 'offset': (5, 16), 'type': 'cell_phone'},
   {'text': '(021)32830431', 'offset': (18, 31), 'type': 'landline_phone'}]

# {'number': '(021)32830431', 'province': '上海', 'city': '上海', 'type': 'landline_phone'}
# {'number': '13288568202', 'province': '广东', 'city': '揭阳',
   'type': 'cell_phone', 'operator': '中国联通'}
```

- 要求输入的文本必须为电话号码字符串，若输入如 “3218177937821332”，很可能造成误识别，即，配合 ```jio.extract_phone_number``` 识别效果佳
- 携号转网后，特定的手机号码会误识别
- 词典定期更新


-----------------------------

## 查找帮助

#### jio.help

若不知道 JioNLP 有哪些功能，可根据命令行提示键入若干关键词做搜索。

```
>>> import jionlp as jio
>>> jio.help()

> please enter keywords in Chinese separated by space:数据增强
> function name ==> jio.BackTranslation
> 回译接口，集成多个公开免费试用机器翻译接 ...
```


-----------------------------

## 公历农历日期互转

#### jio.lunar2solar、jio.solar2lunar

给定公历日期和农历日期，获取其相对应的农历日期和公历日期。若该日期不存在，则报错。

```
>>> import datetime
>>> import jionlp as jio
>>> res = jio.solar2lunar(datetime.datetime(1989, 10, 22))
>>> print('1989-10-22 ==> ', res)
>>> res = jio.lunar2solar(1989, 9, 23, False)
>>> print('1989-9-23 非闰月 ==> ', res)

# 1989-10-22 ==> (1989, 9, 23, False)
# 1989-9-23 非闰月 ==> 1989-10-22 00:00:00
```

- 农历日期存在闰月与非闰月，若指定的闰月不存在，或其它日期不存在，则报错，如 ```jio.solar2lunar(1989, 9, 23, True)``` 1989年农历闰九月廿三，结果为不存在该日期。



