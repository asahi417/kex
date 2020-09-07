# -*- coding: utf-8 -*-
""" UnitTest for keyphrase extraction algorithm in general
 - Japanese
 - verb
"""

import unittest
import keyphraser
from glob import glob
import json
import os


INSTANCES = dict(
    TopicRank=keyphraser.algorithm_instance('TopicRank', language='jp'),
    TextRank=keyphraser.algorithm_instance('TextRank', language='jp'),
    TFIDFBased=keyphraser.algorithm_instance('TFIDFBased', language='jp')
)
INSTANCES_VERB = dict(
    TopicRank=keyphraser.algorithm_instance('TopicRank', add_verb=True, language='jp'),
    TextRank=keyphraser.algorithm_instance('TextRank', add_verb=True, language='jp'),
    TFIDFBased=keyphraser.algorithm_instance('TFIDFBased', add_verb=True, language='jp')
)

TEST_SENT_NO_KW = u"ああああああ。"
TEST_SENT = u"ここに,テストとしてAKB48を置いておきます。あとは東京オリンピックも大事ですね。"
TEST_SENT_EXPECT = ['AKB48',  u"東京オリンピック"]


class TestAlgorithmJP(unittest.TestCase):
    """Test basic algorithm output"""

    @staticmethod
    def get_result(output, key='raw'):
        def tmp(__output):
            if len(__output) == 0:
                return __output
            else:
                return [o[key][0] for o in __output]

        return [tmp(_o) for _o in output]

    def do_test(self, target, expected, algorithm=None, add_verb=False, n=5):
        if algorithm is None:
            if add_verb:
                instances = list(INSTANCES_VERB.values())
            else:
                instances = list(INSTANCES.values())
        else:
            if add_verb:
                instances = [INSTANCES_VERB[algorithm]]
            else:
                instances = [INSTANCES[algorithm]]
        for t, e in zip(target, expected):
            for instance in instances:
                out = instance.extract(t, n)
                out = self.get_result(out, 'raw')
                for a, b in zip(out, e):
                    if not set(a) == set(b):
                        print()
                        print(' - output:', set(a))
                        print(' - expect:', set(b))
                        assert set(a) == set(b)

    def test_numeric_input(self):
        """ numeric input should be ignore and return empty list"""
        target = [[0.1, 100], [0.1], [110], [500, TEST_SENT], [90, TEST_SENT_NO_KW]]
        expected = [[[], []], [[]], [[]], [[], TEST_SENT_EXPECT], [[], []]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TextRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TFIDFBased', add_verb=False, n=5)

    def test_empty(self):
        """ empty input should be ignore and return empty list"""
        target = [[''], [], ['', TEST_SENT], ['', TEST_SENT_NO_KW], [None]]
        expected = [[[]], [], [[], TEST_SENT_EXPECT], [[], []], [[]]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TextRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TFIDFBased', add_verb=False, n=5)

    def single_term_docs(self):
        """ document with single word"""
        target = [
            ['ミンデン＝リュベッケ郡','サー・マイケル・フィリップ・ジャガー']
            ]
        expected = [[['ミンデン＝リュベッケ郡'], ['サー・マイケル・フィリップ・ジャガー']]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TextRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TFIDFBased', add_verb=False, n=5)



    def test_cdots(self):
        """ integrate entity split by `・` and `=` """
        target = [
            ['ミンデン=リュベッケ郡で？ サー・マイケル・フィリップ・ジャガーは？', '“ダン” ダニエル・ジャドソン・キャラハンは？']
            ]
        expected = [[['ミンデン=リュベッケ郡', 'サー・マイケル・フィリップ・ジャガー'], ['ダン', 'ダニエル・ジャドソン・キャラハン']]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TextRank', add_verb=False, n=5)
        self.do_test(target, expected, algorithm='TFIDFBased', add_verb=False, n=5)


    def test_basic(self):
        """ basic test """
        target = [
            [
                u"ウーマンラッシュアワーの村本大輔（38）が24日深夜、ツイッターを更新し、同日に米軍普天間飛行場の名護市辺野古移設をめぐる県民投票が行われた沖縄県を訪問したと報告した。"
                u"村本はこの日、那覇市内で「沖縄県民、笑いにおいで」と名付けた独演会を開催。ツイッターで「先着100名　今夜　村本が突如現れただ面白い話をしお客さんは美味しい"
                u"ご飯を食べながらただ大笑いする夜〜面白いか、面白くないに◯を、どちらでもないはやめてね〜独演会料金はカンパ制」とツイート。インスタグラムで配信も行った。村本は"
                u"「モヤモヤして沖縄に。県民投票の空気を見にきた。ハワイから問題を他人事にできない人が来てるのに、日本の本土の人は、県民投票がなにかすら知らない。東京中心で作られる"
                u"メディアとはなんだろう」と、沖縄以外の日本全国での関心の低さを嘆いた。その上で「いま平和で幸せな人からしたら、沖縄ってなんだろう。濃いい時間だった」（コメントは原文のまま）と意味深にツイートした。"
            ]
        ]
        print()
        print('*** without verb ***')
        for t in target:
            for instance in INSTANCES.values():
                out = instance.extract(t, count=5)
                for _o in out:
                    assert len(_o) == 5
                    print(instance.__module__, [k['raw'][0] for k in _o])
                    assert '行わ' not in [k['raw'][0] for k in _o]

        print('*** with verb ***')
        for t in target:
            for instance in INSTANCES_VERB.values():
                out = instance.extract(t, count=10)
                for _o in out:
                    print(instance.__module__, [k['raw'][0] for k in _o])
                    assert '行わ' in [k['raw'][0] for k in _o]
        print()


if __name__ == "__main__":
    unittest.main()
