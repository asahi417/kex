""" Preprocess raw document to get sequence of phrase and token [Japanese] """

import re
import MeCab
import os
from .basic_modules import CleaningStr, CandidatePhrase


POS_MAPPER = {
    "名詞": "NOUN",
    "形容詞": "ADJ",
    "動詞": "VERB",
    "RANDOM": "RANDOM"
}


def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def tokeinzer(sentence,
              mecab_tagger):
    """ MeCab wrapper: (token, 9: 0(POS), 6(basic form), 1(raw form))

     Usage
    --------------
    >>> import MeCab
    >>> tagger = MeCab.Tagger()
    >>> tokeinzer("今日は華金だからビールが飲めるぞー", tagger)
        [['NOUN', '今日', '今日'],
         ['RANDOM', '*', 'は'],
         ['NOUN', '華', '華'],
         ['NOUN', '金', '金'],
         ['RANDOM', '*', 'だ'],
         ['RANDOM', 'から', 'から'],
         ['NOUN', 'ビール', 'ビール'],
         ['RANDOM', '*', 'が'],
         ['RANDOM', '飲める', '飲める'],
         ['RANDOM', '*', 'ぞ'],
         ['NOUN', '*', 'ー']]
    """

    def map_pos(pos):
        if pos not in POS_MAPPER.keys():
            return POS_MAPPER['RANDOM']
        else:
            return POS_MAPPER[pos]

    def cleansing(token):
        # sometimes MeCab returns multiple tokens or token with 。 eg) `DDR SSD`, `結婚。`
        token = token.replace("。", "")

        # if single character and not kanji, reject
        is_single_char = len(token) == 1
        kanji_range = "一-龠"
        is_kanji = re.search(r"[%s]" % kanji_range, token)
        if is_single_char and not is_kanji:
            return '*'
        else:
            return token

    def formatting(_list, raw):
        basic_form = cleansing(_list[6])
        if is_digit(raw) or is_digit(basic_form):
            pos = POS_MAPPER['RANDOM']
        elif basic_form == '':
            pos = POS_MAPPER['RANDOM']
        else:
            pos = map_pos(_list[0])
        return [pos, basic_form, raw]

    parsed_sentence = mecab_tagger.parse(sentence).split("\n")
    out = [formatting(s.split("\t")[1].split(","), s.split("\t")[0]) for s in parsed_sentence if "\t" in s]
    return out


class Processing:
    """ Pipeline of cleaning filter, MeCab object to get nlp attribution (stem, lemma, pos), and phrasing.

     Usage
    -----------------
    >>> sample =
    'DVDレンタルや書籍販売のTSUTAYA（ツタヤ、東京）が展開する動画配信サービス「TSUTAYA\u3000TV」で、全作品を見放題であるかのように宣伝したのは虚偽であり、
    景品表示法違反（優良誤認）に当たるとして、消費者庁は22日、ツタヤに課徴金1億1753万円の納付命令を出した。同庁表示対策課によると、ツタヤは2016〜18年、
    ホームページや動画投稿サイト「ユーチューブ」で、「動画見放題」「動画見放題＆定額レンタル8」「TSUTAYA\u3000プレミアム」のプランについて、全ての作品が見放題であるように宣伝
    しかし、実際に見放題なのは全作品のうち最大で27％だった。'
    >>> pro = Processing(path_to_neologd='/usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    >>> result_dict = pro(sample)
    >>> result_dict.keys()
        dict_keys(['DVDレンタル', '書籍販売', 'TSUTAYA', '東京', '展開', '動画配信サービス', 'TSUTAYA TV', '作品', '放題',
        '宣伝', '虚偽', '景品表示法違反', '優良誤認', '消費者庁', '課徴金1億', '納付命令', '同庁表示対策', 'ホームページ',
        '動画投稿サイト', 'You Tube', '動画見放題', '定額レンタル', 'TSUTAYAプレミアム', 'プラン', '全て', 'うち最大'])
    >>> result_dict['DVDレンタル'
        {'count': 1,
         'lemma': ['DVDレンタル'],
         'offset': [[0, 1]],
         'pos': 'NOUN NOUN',
         'raw': ['DVDレンタル'],
         'stemmed': 'DVDレンタル'}

    """


    def __init__(self,
                 path_to_neologd: str=None,
                 add_verb: bool=False):

      # cleaning filter instance
        self.__clean_str = CleaningStr(language='jp')

        # setup spacy MecCab
        self.__tagger = MeCab.Tagger()
        if path_to_neologd is None:
            f = os.popen('echo `mecab-config --dicdir`"/mecab-ipadic-neologd"')
            path_to_neologd = f.read().replace('\n', '')

        if os.path.exists(path_to_neologd):
            self.__tagger = MeCab.Tagger("-d %s" % path_to_neologd)
        else:
            self.__tagger = MeCab.Tagger()
        self.__tagger.parse('テスト')

        # candidate phrasing
        self.__add_verb = add_verb

    def __call__(self,
                 single_doc: str,
                 return_token: bool = False):

        single_doc = single_doc.replace('\u3000', ' ')
        single_doc = self.__clean_str.process(single_doc)
        pos_token = tokeinzer(single_doc, self.__tagger)

        phrase_instance = CandidatePhrase(join_without_space=True,
                                          add_verb=self.__add_verb)

        cleaned_str_tokenize = []
        n = 0
        for n, (_pos, _token, _raw) in enumerate(pos_token):
            if _token == '*':
                _token = _raw
            # print(_token, _raw)

            # string filter
            # _token = self.__clean_str.numeric(_token, language='jp')

            cleaned_str_tokenize.append(_token)
            # print(_token, _raw)
            phrase_instance.add(raw=_raw,
                                lemma=_token,
                                stemmed=_token,
                                pos=_pos,
                                offset=n)

        # to finalize the phrase in the end of sentence
        phrase_instance.add(raw='.',
                            lemma='.',
                            stemmed='.',
                            pos='PUNCT',
                            offset=n+1)

        if return_token:
            return phrase_instance.phrase, cleaned_str_tokenize
        else:
            return phrase_instance.phrase
