# -*- coding=utf-8 -*-

import os
import sys
import numpy as np
import numpy.ctypeslib as npct
import ctypes
import pdb
import json
# import jionlp as jio
# from jiojio.tag_words_converter import tag2word


dir_path = '/home/ubuntu/github/jiojio/jiojio/jiojio_cpp'
feature_extractor = ctypes.cdll.LoadLibrary(
    os.path.join(dir_path, 'build', 'libfeatureEx.so'))
feature_extractor = ctypes.PyDLL(os.path.join(dir_path, 'build', 'libfeatureEx.so'))
get_feature_c = feature_extractor.getFeature
get_feature_c.argtypes = [
    ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int, ctypes.py_object, ctypes.py_object]
get_feature_c.restype = ctypes.py_object

unigram = set(["nc", 'kl', "今天", "kfaewfu", "3rqiu4", "R$2这", "总统府", "挺好的", 'nc\x00'])
text = '今天天气挺好的啊，ncfds。'

# text = "一切存在之物，各自有其存在之本质，所谓本质（essence），即物之为物，所必具之固有性，缺此要素，则不成其为同类之物也，故本质常含有普遍性，必然性，而为某事某物共通之特质。"
unigram = set(["天气", "今天", "中国", "美国", "总统", "总统府"])
bigram = set(["美国.总统", "天气.晴朗", "美国.总统府"])
# unigram = list(["天气", "今天", "中国"])

with open('/home/ubuntu/datasets/unigram.json', 'r', encoding='utf-8') as fr:
    unigram = set(json.load(fr))
# with open('/home/ubuntu/datasets/bigram.json', 'r', encoding='utf-8') as fr:
#     bigram = set(json.load(fr))
# print(text[25])
# pdb.set_trace()
unigram = {'任何', '方', '无产阶级', '意义', '场', '方便', '不仅', '基础', '交流', '价值', '几个', '着',
           '责任', '还有', '新', '下去', '资料', '每个', '同样', '成果', '多年', '需', '或者', '那样',
           '小时', '从', '尽管', '形式', '冠军', '他', '是', '南朝鲜', '不要', '利用', '不得', '还',
           '工程', '成为', '队伍', '突出', '保证', '第二', '而', '稳定', '付', '回答', '各地', '反映',
           '建筑', '国内', '局面', '军事', '表演', '热烈', '那个', '拉', '胡鞍钢', '讲', '其中', '前',
           '一家', '花', '作用', '而且', '目标', '单位', '越', '效果', '靠', '经销', '维修', '面对',
           '决定', '货', '元', '机会', '节约', '一代', '来自', '斤', '便', '既', '举行', '书', '交通',
           '肯定', '之后', '虽', '错误', '平均', '道德', '事业', '有人', '担', '表现', '达', '分钟',
           '无', '人员', '优势', '工艺', '信息', '中国', '大', '计划', '来信', '给予', '奋斗', '商店',
           '几十', '带来', '革命', '天津', '这么', '一直', '头', '自治区', '追求', '互相', '管理',
           '为了', '香港', '乐', '认为', '美', '原因', '代表团', '三十', '反对', '真理', '寨', '了解',
           '田', '世界', '处', '小', '厂', '晚上', '活', '依靠', '所有', '逐步', '社', '体操', '心',
           '到', '纪录', '明白', '完成', '困境', '双方', '几乎', '底', '主动', '而是', '秒', '得到',
           '建', '事件', '口', '五十', '向', '一致', '提', '不', '面积', '武装', '书记', '内', '南京',
           '出口', '就业', '次', '先生', '改革开放', '著名', '大队', '委员会', '帝国主义', '重点',
           '指', '领导人', '确定', '当', '通知', '项目', '坚决', '教师', '调查', '球', '学会',
           '运动员', '国民经济', '孩子', '石油', '这些', '深刻', '圆', '去', '正在', '政策', '此',
           '找', '强', '十三', '低', '亩', '行动', '坐', '比较', '枚', '原来', '边', '报', '记',
           '达到', '协助', '各个', '非洲', '更', '然而', '改变', '全国', '数', '有些', '组织', '队员',
           '日', '取得', '方法', '班', '这里', '南', '跟', '再', '不管', '于是', '除了', '人才', '该',
           '道', '倍', '声', '结构', '民主', '似乎', '设计', '特点', '按照', '京', '物', '损失',
           '原料', '三', '服务', '基础上', '关于', '局', '引进', '推动', '几', '这位', '以上', '能力',
           '一些', '自然', '件', '母亲', '不正之风', '对外', '具有', '因而', '又', '中央', '战',
           '青年', '回', '重视', '地', '这样', '规范', '工作者', '卫生', '门', '本市', '永远', '做',
           '重要', '环境', '旧', '周', '转变', '既然', '绝对真理', '国情', '点', '坚持', '粮食',
           '干', '项', '经', '两', '不少', '六', '吧', '个人', '大力', '还是', '恢复', '有', '树立',
           '乱', '某些', '集中', '矛盾', '大家', '旅游', '法', '曾', '哼', '实现', '去年', '产品',
           '动', '发生', '工人阶级', '影响', '大型', '不可', '物质', '才', '类', '满意', '领导',
           '进口', '啊', '按', '决议', '那么', '决心', '好', '负责', '叫', '的', '调整', '解放',
           '真', '干部', '工业', '使用', '保持', '很快', '回来', '面向', '搞', '重大', '下来', '就',
           '对于', '被', '政府', '到了', '下', '应该', '接受', '报道', '日本', '举办', '文化', '长期',
           '半', '呢', '优质', '不幸', '津', '题材', '同', '下午', '一天', '四十', '二十', '以后',
           '意见', '问', '青少年', '地区', '中心', '经验', '金牌', '大姐', '伟大', '啤酒', '提出',
           '里', '九月', '语言', '它', '您', '方针', '措施', '我们', '家里', '共同', '行', '男子',
           }
print(len(text))
unigram = set()

unigram.add('据')
unigram.add('nc')
unigram.add('nc\x00')
unigram.add('kl')
print('length of unigram: ', len(unigram), 'nc' in unigram)
# pdb.set_trace()
res = get_feature_c(8, text, len(text), unigram, bigram)

print(res)
print(type(res[0]))
sys.exit()