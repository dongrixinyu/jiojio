# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

"""
方法说明：
    0、该方法用在训练数据集处理、推断文本预处理阶段，可以减少模型复杂度，减少罕见字符特征的训练，提高模型的稳定性。
    1、该方法主要正规化 数字、英文字母，全半角空格等符号；正规化无多个语言功能的标点符号；
        正规化罕见字符与、日、俄等假名、西里尔字母等信息。
    2、该方法强调处理速度，以最大速率对文本进行归一化，耗时评估见下。

耗时评估：
    0、调用方式在 PreProcessor 类中的默认调用方法 __call__ 中，test 方法为测速方法
    1、在 __call__ 中直接调用 maketrans 速度比 调用 re 中的 sub 与 python 默认
        replace 方法快大约一个数量级。
    2、在 __call__ 中调用子方法函数 也会对耗时有大约 3%~6% 的耗时增加，因此建议减少
        函数的多次调用。
    3、maketrans 方法应当尽量将多个合并成一个，减少对字符串的遍历。

规则说明：
    1、某些字母和数字会引起混淆，如 “一九八Ｏ” 中，实际上以全角字母 “O” 替代。因此字母和数字不可归一化替代。
    2、
"""


import re


SINGLE_CHINESE_FAMILY_NAME = \
    '赵李吴郑王冯陈褚蒋沈韩杨朱秦尤许何吕张孔曹严魏陶姜戚邹喻窦潘葛奚范彭郎鲁韦' \
    '俞袁酆鲍史廉岑薛倪滕殷罗郝邬傅卞康伍卜顾孟穆萧尹姚邵' \
    '湛汪祁禹狄臧宋茅董梁杜阮闵贾娄郭盛刁钟徐邱骆' \
    '蔡樊凌霍虞柯昝卢莫裘缪丁贲邓郁崔龚嵇邢裴翁荀於惠甄' \
    '芮羿储靳汲邴糜弓隗侯宓蓬郗仲伊栾钭刘詹' \
    '韶郜黎蓟薄蒲邰鄂蔺乔胥莘翟谭贡逄姬冉郦雍郤璩' \
    '濮扈冀郏尚晏瞿阎慕茹宦艾易慎戈廖庾暨衡耿弘匡寇禄阙' \
    '殳沃夔厍聂晁敖融訾阚饶毋乜鞠巢蒯後竺逯桓' \
    '仉晋楚闫汝鄢涂钦缑亢牟佘佴赏谯笪佟'

TWO_CHAR_CHINESE_FAMILY_NAME = \
    '万俟|司马|上官|欧阳|夏侯|诸葛|闻人|东方|赫连|皇甫|尉迟|公羊|澹台|公冶|宗政|濮阳|淳于|单于|太叔|申屠|' \
    '公孙|仲孙|轩辕|令狐|钟离|宇文|长孙|慕容|鲜于|闾丘|司徒|司空|亓官|司寇|子车|颛孙|端木|巫马|公西|漆雕|' \
    '乐正|壤驷|公良|拓跋|夹谷|宰父|谷梁|段干|百里|东郭|南门|呼延|羊舌|微生|梁丘|左丘|东门|西门|南宫|第五'

CHINESE_FAMILY_NAME = '(' + '|'.join(SINGLE_CHINESE_FAMILY_NAME) + \
                      '|' + TWO_CHAR_CHINESE_FAMILY_NAME + ')'


class PreProcessor(object):
    """清洗字符串，准备用户词典"""

    def __init__(self, convert_num_letter=True, normalize_num_letter=False,
                 convert_exception=True):

        # 检查是否包含中文字符
        self.chinese_char_pattern = re.compile('[一-龥]')
        self.num_pattern = re.compile(
            r'^(\d+(,\d+)?(\.\d+)?(万|亿|万亿|万千|千万|千|点|亿千|兆)|[\d]+(%|％)?|'\
            r'([\d]+)?\.([\d]+)?(%|％)?|'\
            r'[\d]+\:[\d]+|'\
            r'[零一二三四五六七八九十百千万亿]{3,9})$')
        self.pure_num_pattern = re.compile(
            r'^([\d]+(%|％)?|'\
            r'([\d]+)?\.([\d]+)?(%|％)?)$')
        self.percent_num_pattern = re.compile('(百分之)')

        # 检测是否为人名正则
        self.chinese_family_name = re.compile(CHINESE_FAMILY_NAME)
        self.two_char_chinese_family_name = re.compile(
            '(' + TWO_CHAR_CHINESE_FAMILY_NAME + ')')

        # 检查应当剔除的时间的正则，这些由组合特征学习得到，不需要参与 unigrams
        self.time_pattern = re.compile(
            r'\d年|\d月(份)?|\d日|\d(小)?时|\d分|\d秒')

        # 预处理参数，用于控制预处理方式
        self.convert_num_letter = convert_num_letter
        self.normalize_num_letter = normalize_num_letter
        self.convert_exception = convert_exception

        # 归一化标点符号，将确定性的标点进行替换，而非确定性标点，
        # 如 “.”不仅仅可以作为句子结尾，还可作为数字小数点，因此不可进行标点归一化
        punctuation = '。，、；！？：“”—《》（）…'  # 此标点可能存在问题，带接续性语义的标点会造成错误
        punctuation = '。；！？…'  # 不会造成任何干扰的标点字符串
        # “：” 可能存在出现在时间中，造成干扰
        # “、” 可能存在小数中，如“0、932”
        # “，” 可能存在数字中，如“11，320.00”
        # punctuation = '。'

        # 归一化字符，数字、英文字母 -> 7\Z
        num = '0123456789几二三四五六七八九十千万亿兆零１２３４５６７８９０①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳' \
              '⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩'  # 缺 百、一
        letter = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ' \
                 'ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ' \
                 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \
                 'abcdefghijklmnopqrstuvwxyz' \
                 '⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵' \
                 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ' \
                 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ'
        space = '　'
        converted_space = ' '
        self.num_letter_space_punc_norm_translation = str.maketrans(
            num + letter + space + punctuation,
            '7' * len(num) + 'Z' * len(letter) + converted_space + len(punctuation) * '。')

        # 归一化非异常、罕见字符 -> 井字符 #

        # 归一化异常字符 -> 日语假名 ん
        ASCII_EXCEPTION_PATTERN = '[^\x09-\x0d\x20-\x7e\xa0£¥©®°±×÷]'
        UNICODE_EXCEPTION_PATTERN = '[^‐-”•·・…‰※℃℉Ⅰ-ⅹ①-⒛\u3000-】〔-〞㈠-㈩一-龥﹐-﹫！-～￠￡￥]'
        EXCEPTION_PATTERN = ASCII_EXCEPTION_PATTERN[:-1] + UNICODE_EXCEPTION_PATTERN[2:]
        self.exception_token_pattern = re.compile(EXCEPTION_PATTERN)

        # 将 特殊的数字、字母转换为 ascii 编码
        num = '０１２３４５６７８９' \
              '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳' \
              '⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇' \
              '⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛' \
              '㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩'
        # 由于 `⑪` 等字符按实转换会得到 `11` 两个字符，因此，将其转换为 `9`
        converted_num = '01234567891234567899999999999912345678999999999999123456789999999999991234567899'
        letter = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ' \
                 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ' \
                 '⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵' \
                 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ' \
                 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ'
        converted_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
                           'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.num_letter_punc_token_translation = str.maketrans(
            num + letter + space + punctuation,
            converted_num + converted_letter + converted_space + len(punctuation) * '。')

    def __call__(self, text):
        if self.normalize_num_letter:
            text = text.translate(self.num_letter_space_punc_norm_translation)
        elif self.convert_num_letter:
            text = text.translate(self.num_letter_punc_token_translation)

        if self.convert_exception:
            text = self.exception_token_pattern.sub('ん', text)

        return text

    def check_chinese_char(self, text):
        if text == '':
            return False

        if self.chinese_char_pattern.search(text):
            return True

        return False

    def check_num(self, text):
        if text == '':
            return False

        if self.num_pattern.search(text) is not None:
            return True
        return False

    def check_chinese_name(self, text):
        text_length = len(text)
        if text_length <= 1:  # 非人名
            return False

        if text_length >= 5:  # 非人名
            return False

        if text_length == 4:
            # 4 字人名，其中包括两种情况：
            # 1、姓氏为二字，如 “欧阳”
            if self.chinese_family_name.search(text[0]) is not None \
                    and self.chinese_family_name.search(text[1]) is not None:
                return True

            # 2、首二字为单字姓氏，如父母姓氏的组合：“刘王晨曦”
            if self.two_char_chinese_family_name.search(text[:2]) is not None:
                return True

            return False

        if text_length == 3:
            # 3 字人名
            # 1、首字为姓氏，如 “张”
            if self.chinese_family_name.search(text[0]) is not None:
                return True

            # 2、姓氏为二字，如 “上官”
            if self.two_char_chinese_family_name.search(text[:2]) is not None:
                return True

            return False

        if text_length == 2:
            if self.chinese_family_name.search(text[0]) is not None:
                return True

            return False

    def cleansing_unigram(self, text):
        """ 输入一个 ungram，判断其是否符合进入 unigram 词典集。
        Return:
            (bool): 是意味符合 unigram 要求，否则将该词串剔除
        """

    def normalize_num_letter(self, text):
        text = text.translate(self.letter_translation)
        text = text.translate(self.num_translation)
        return text

    def convert_full2half(self, text):
        return text.translate(self.full_angle_translation)

    def convert_num_letter(self, text):
        return text.translate(self.num_letter_token_translation)

    def convert_space(self, text):
        return text.translate(self.space_translation)

    def convert_exception(self, text):
        return self.exception_token_pattern.sub('ん', text)

    def _test(self, text, convert_num_letter=True, normalize_num_letter=True,
             convert_exception=True):
        if normalize_num_letter:
            text = text.translate(self.num_letter_space_norm_translation)
        elif convert_num_letter:
            # 若正规化所有数字、字母，则全半角、符号等失去意义
            text = text.translate(self.num_letter_punc_token_translation)

        if convert_exception:
            text = self.exception_token_pattern.sub('ん', text)

        return text

if __name__ == '__main__':

    import pdb
    # from jiojio import TimeIt
    import jionlp as jio
    text = '''0123456789几二三四五六七八九十千万亿兆零１２３４５６７８９０①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳' \
              '⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩'  # 缺 百、一
        letter = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ' \
                 'ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ' \
                 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
                 '⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵' \
                 'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ' \
                 'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ'''
    # text = "\x09-\x0d\x20-fdf3qr34f4srf45\x7e\xa0£¥©®°±×÷]㐀䶵䶵㐀䶵㐀䶵㐀㐀㐀㐀-䶵'"\
    #        "UNICODE_EXCEPTION_PATTERN = '[^‐-”•·・…‰※℃℉Ⅰ-ⅹ①-⒛\u3000-】〔-〞㈠-㈩一-龥﹐-﹫！-～￠￡￥]'"
    text = '第１４分钟，刚刚恢复到较佳状态的㐀䶵䶵㐀䶵㐀䶵㐀㐀㐀㐀-䶵守门员朱力飞身挡出一记７米球后，以２０.５比２５.５落后５分的天津队'\
        '士气大振，利用对手失误增多之机连打反击，１１号"拚命三郎"翟文明表现出色，利用快攻和准确的９米线外远'\
        '射，连得５分。'
    pre_processor = PreProcessor()

    pdb.set_trace()
    with jio.TimeIt('re func'):
        for i in range(100000):
            res = pre_processor(text, normalize_num_letter=False)
            # pdb.set_trace()

    with jio.TimeIt('str func'):
        for i in range(100000):
            res = pre_processor._test(text, normalize_num_letter=False)  # , convert_exception=False)

            # pdb.set_trace()
    length = len(text) * 100000
    pdb.set_trace()
