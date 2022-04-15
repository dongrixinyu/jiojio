# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import re
import pdb


__all__ = ['Extractor']


class Extractor(object):
    """ 规则抽取器 """
    def __init__(self):
        self.email_pattern = re.compile(
            r'(?<=[^0-9a-zA-Z.\-])' \
            r'([a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+(?:\.[a-zA-Z0-9_.-]+)*\.[a-zA-Z0-9]{2,6})' \
            r'(?=[^0-9a-zA-Z.\-])')

        self.url_pattern = re.compile(
            r'(?<=[^.])((?:(?:https?|ftp|file)://|(?<![a-zA-Z\-\.])www\.)' \
            r'[\-A-Za-z0-9\+&@\(\)#/%\?=\~_|!:\,\.\;]+[\-A-Za-z0-9\+&@#/%=\~_\|])' \
            r'(?=[<\u4E00-\u9FA5￥，。；！？、“”‘’>（）—《》…● \t\n])')

        ip_single = r'(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)'
        self.ip_address_pattern = re.compile(
            ''.join([r'(?<=[^0-9])(', ip_single, r'\.', ip_single, r'\.',
                     ip_single, r'\.', ip_single, ')(?=[^0-9])']))

        self.id_card_pattern = re.compile(
            r'(?<=[^0-9a-zA-Z])' \
            r'((1[1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4]|6[1-5]|71|81|82|91)' \
            r'(0[0-9]|1[0-9]|2[0-9]|3[0-4]|4[0-3]|5[1-3]|90)' \
            r'(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-3]|5[1-7]|6[1-4]|7[1-4]|8[1-7])' \
            r'(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])' \
            r'\d{3}[0-9xX])' \
            r'(?=[^0-9a-zA-Z])')

        self.cell_phone_pattern = re.compile(
            r'(?<=[^\d])(((\+86)?([- ])?)?((1[3-9][0-9]))([- ])?\d{4}([- ])?\d{4})(?=[^\d])')
        self.landline_phone_pattern = re.compile(
            r'(?<=[^\d])(([\(（])?0\d{2,3}[\)） —-]{1,2}\d{7,8}|\d{3,4}[ -]\d{3,4}[ -]\d{4})(?=[^\d])')

    @staticmethod
    def _extract_base(pattern, text, with_offset=False, typing='none'):
        """ 正则抽取器的基础函数

        Args:
            pattern(re.compile): 正则表达式对象
            text(str): 字符串文本
            with_offset(bool): 是否携带 offset （抽取内容字段在文本中的位置信息）
            type(str): 抽取的字段类型

        Returns:
            list: 返回结果

        """
        if with_offset:
            results = [{'text': item.group(1),
                        'offset': (item.span()[0] - 1, item.span()[1] - 1),
                        'type': typing}
                       for item in pattern.finditer(text)]
        else:
            results = [item.group(1) for item in pattern.finditer(text)]

        return results

    def extract_email(self, text, detail=False):
        """ 提取文本中的 E-mail

        Args:
            text(str): 字符串文本
            detail(bool): 是否携带 offset （E-mail 在文本中的位置信息）

        Returns:
            list: email列表

        """
        text = ''.join(['#', text, '#'])
        results = self._extract_base(
            self.email_pattern, text, with_offset=detail, typing='email')

        if not detail:
            return results
        else:
            detail_results = list()
            for item in results:
                detail_results.append(item)
            return detail_results

    def extract_id_card(self, text, detail=False):
        """ 提取文本中的 ID 身份证号

        Args:
            text(str): 字符串文本
            detail(bool): 是否携带 offset （身份证在文本中的位置信息）

        Returns:
            list: 身份证信息列表

        """
        text = ''.join(['#', text, '#'])
        return self._extract_base(
            self.id_card_pattern, text, with_offset=detail, typing='id')

    def extract_ip_address(self, text, detail=False):
        """ 提取文本中的 IP 地址

        Args:
            text(str): 字符串文本
            detail(bool): 是否携带 offset （IP 地址在文本中的位置信息）

        Returns:
            list: IP 地址列表

        """
        text = ''.join(['#', text, '#'])
        return self._extract_base(
            self.ip_address_pattern, text, with_offset=detail, typing='ip')

    def extract_phone_number(self, text, detail=False):
        """从文本中抽取出电话号码

        Args:
            text(str): 字符串文本
            detail(bool): 是否携带 offset （电话号码在文本中的位置信息）

        Returns:
            list: 电话号码列表

        """
        text = ''.join(['#', text, '#'])
        cell_results = self._extract_base(
            self.cell_phone_pattern, text, with_offset=detail, typing='tel')
        landline_results = self._extract_base(
            self.landline_phone_pattern, text, with_offset=detail, typing='tel')

        if not detail:
            return cell_results + landline_results
        else:
            detail_results = list()
            for item in cell_results:
                item.update({'type': 'cell_phone'})
                detail_results.append(item)
            for item in landline_results:
                item.update({'type': 'landline_phone'})
                detail_results.append(item)
            return detail_results

    def extract_url(self, text, detail=False):
        """提取文本中的url链接

        Args:
            text(str): 字符串文本
            detail(bool): 是否携带 offset （URL 在文本中的位置信息）

        Returns:
            list: url列表

        """
        text = ''.join(['￥', text, '￥'])  # 因 # 可出现于 url

        return self._extract_base(
            self.url_pattern, text, with_offset=detail, typing='url')
