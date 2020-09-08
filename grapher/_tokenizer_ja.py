""" MeCab wrapper for Japanese tokenizer"""
import os

import MeCab

__all__ = 'TokenizerJa'


class TokenizerJa:
    """ MeCab wrapper """

    POS_MAPPER = {
        "名詞": "NOUN",
        "形容詞": "ADJ",
        "動詞": "VERB",
        "RANDOM": "RANDOM"
    }

    def __init__(self, neologd: bool = True):
        # setup spacy MecCab
        self.__tagger = MeCab.Tagger()
        if neologd:
            f = os.popen('echo `mecab-config --dicdir`"/mecab-ipadic-neologd"')
            path_to_neologd = f.read().replace('\n', '')
            if os.path.exists(path_to_neologd):
                self.__tagger = MeCab.Tagger("-d %s" % path_to_neologd)
            else:
                self.__tagger = MeCab.Tagger("")
        self.__tagger.parse('テスト')

    def tokenize(self, sentence):

        def is_digit(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        def map_pos(pos):
            if pos not in self.POS_MAPPER.keys():
                return self.POS_MAPPER['RANDOM']
            else:
                return self.POS_MAPPER[pos]

        def formatting(_list, raw):
            basic_form = _list[6].replace("。", "")
            if is_digit(raw) or is_digit(basic_form):
                pos = self.POS_MAPPER['RANDOM']
            elif basic_form == '':
                pos = self.POS_MAPPER['RANDOM']
            else:
                pos = map_pos(_list[0])
            return [pos, basic_form, raw]

        parsed = self.__tagger.parse(sentence)
        if parsed is None:
            return None
        parsed_sentence = parsed.split("\n")
        out = [formatting(s.split("\t")[1].split(","), s.split("\t")[0]) for s in parsed_sentence if "\t" in s]
        return out
