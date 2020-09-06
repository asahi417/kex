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
    TextRank=keyphraser.algorithm_instance('TextRank', language='jp')
)
INSTANCES_VERB = dict(
    TopicRank=keyphraser.algorithm_instance('TopicRank', add_verb=True, language='jp'),
    TextRank=keyphraser.algorithm_instance('TextRank', add_verb=True, language='jp')
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
                # if not out == e:
                #     print(out, e)
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
        # self.do_test(target, expected, add_verb=True)

    def test_empty(self):
        """ empty input should be ignore and return empty list"""
        target = [[''], [], ['', TEST_SENT], ['', TEST_SENT_NO_KW], [None]]
        expected = [[[]], [], [[], TEST_SENT_EXPECT], [[], []], [[]]]
        self.do_test(target, expected, algorithm='TopicRank', add_verb=False, n=5)
        # self.do_test(target, expected, add_verb=True)

    def test_basic(self):
        """ basic test """
        target = [
            [
                u"毎月勤労統計の調査方法を巡り、厚生労働省の藤沢勝博政策統括官は25日の衆院予算委員会で、同省担当者が有識者検討会座長に送った2015年9月4日のメールの「官邸関係者」は、当時の内閣参事官と認めた。調査方法変更については、官邸の意向を重ねて否定した。野党は調査方法変更に官邸の圧力があったと主張。「国会答弁は虚偽の可能性が高い」と猛反発している。当時の内閣参事官は、横幕章人・現厚労省会計課長。　藤沢氏は、横幕氏が厚労省の聴取に「内容がテクニカルと思われたので中江元哉・元首相秘書官には報告していないのではないか」と述べていることを明らかにした。",
                u"２５日午前５時頃、ＪＲ中央線の神田―四ツ谷駅間（東京都）で停電が発生した。これにより、中央・総武線各駅停車が三鷹―西船橋駅間の上下線で約４時間、中央線快速は東京―新宿駅間の上下線で約４時間半、運転を見合わせるなど、上下線で１７７本が運休、９１本に遅れが出た。朝の通勤・通学時間帯を直撃し、２８万人に影響した。ＪＲ東日本と警視庁麹町署によると、停電の原因は、中央・総武線の水道橋駅付近で、信号や駅の電光掲示板などの施設に電気を送る送電用ケーブルから出火し、近くの資機材を焼いたためとみられる。火災は約１時間２０分後に消し止められた。同日未明に水道橋―飯田橋駅付近で橋桁の改良工事が行われており、ＪＲ東日本などは、溶接や研磨の作業で火花が飛んだ可能性もあるとみて調べている。停電により、中央・総武線の西船橋―千葉駅間、中央線快速の新宿―高尾駅間でも一時、上下線で運転を見合わせた。各駅は通勤・通学の乗客で大混雑し、御茶ノ水駅や水道橋駅などでは、ホームから人があふれるのを避けるため、改札口で入場を規制した。私鉄にも影響が出た。東京メトロは東西線の中野―三鷹駅間、西船橋―津田沼駅間で直通運転を中止。ＪＲからの振り替え輸送の乗客で、同線や丸ノ内線の各駅が混雑。西武新宿線や京王線などにも遅れが出た。同日午前１１時時点で混雑などによるけが人は確認されていない。この日は国公立大２次試験の前期日程で、中央・総武線の沿線大学の受験生が試験時間に間に合わない事態も起きた。一部の大学は、試験開始時間を１時間程度繰り下げる対応を取った。",
                u"ウーマンラッシュアワーの村本大輔（38）が24日深夜、ツイッターを更新し、同日に米軍普天間飛行場の名護市辺野古移設をめぐる県民投票が行われた沖縄県を訪問したと報告した。村本はこの日、那覇市内で「沖縄県民、笑いにおいで」と名付けた独演会を開催。ツイッターで「先着100名　今夜　村本が突如現れただ面白い話をしお客さんは美味しいご飯を食べながらただ大笑いする夜〜面白いか、面白くないに◯を、どちらでもないはやめてね〜独演会料金はカンパ制」とツイート。インスタグラムで配信も行った。村本は「モヤモヤして沖縄に。県民投票の空気を見にきた。ハワイから問題を他人事にできない人が来てるのに、日本の本土の人は、県民投票がなにかすら知らない。東京中心で作られるメディアとはなんだろう」と、沖縄以外の日本全国での関心の低さを嘆いた。その上で「いま平和で幸せな人からしたら、沖縄ってなんだろう。濃いい時間だった」（コメントは原文のまま）と意味深にツイートした。"
            ]
        ]
        print()
        print('*** with verb ***')
        for t in target:
            for instance in INSTANCES.values():
                out = instance.extract(t, 5)
                for _o in out:
                    assert len(_o) == 5
                    print(instance.__module__, [k['raw'][0] for k in _o])
        print('*** with verb ***')
        for t in target:
            for instance in INSTANCES_VERB.values():
                out = instance.extract(t, 5)
                for _o in out:
                    assert len(_o) == 5
                    print(instance.__module__, [k['raw'][0] for k in _o])
        print()


if __name__ == "__main__":
    unittest.main()
