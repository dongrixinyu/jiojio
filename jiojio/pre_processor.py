# -*- coding=utf-8 -*-

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

"""


import re


class PreProcessor(object):
    """清洗字符串，准备用户词典"""

    def __init__(self, convert_num_letter=True, normalize_num_letter=False,
                 convert_exception=True):
        # 预处理参数，用于控制预处理方式
        self.convert_num_letter = convert_num_letter
        self.normalize_num_letter = normalize_num_letter
        self.convert_exception = convert_exception

        # 归一化标点符号，将确定性的标点进行替换，而非确定性标点，
        # 如 “.”不仅仅可以作为句子结尾，还可作为数字小数点，因此不可进行标点归一化
        punctuation = '。，、；！？：“”—《》（）…'  # 此标点可能存在问题，带接续性语义的标点会造成错误
        punctuation = '。，、；！？：…'

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
