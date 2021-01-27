""" tokenizer/stemmer/PoS tagger/phrase constructor """
import re
from typing import List
from itertools import chain

import nltk
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag

from ._stopwords import get_stopwords_list


__all__ = 'PhraseConstructor'


class Phrase:
    """ Data structure for phrases, which follows regx [Adjective]*[Noun|proper noun]+

    The structure contains a main dictionary, in which key is the stemmed candidates (so, phrases which share same
    stemmed form are regarded as same phrase)
        * pos: pos
        * stemmed: stemmed form
        * raw: list of raw phrase
        * offset: offset
    Also exclude token in given stop word list.
    """

    def __init__(self,
                 join_without_space: bool = False,
                 maximum_word_number: int = None,
                 maximum_char_number: int = None):
        """ Data structure for phrases, which follows regx [Adjective]*[Noun|proper noun]+

         Parameter
        -------------------
        join_without_space: join token without halfspace to construct a phrase (eg, Japanese)
        """

        self.__content = dict()
        self.__initialize_list()
        self.__joiner = '' if join_without_space else ' '
        self.__maximum_word_number = maximum_word_number
        self.__maximum_char_number = maximum_char_number

    @property
    def phrase(self):
        return self.__content

    def __initialize_list(self):
        self.__tmp_phrase_raw = []
        self.__tmp_phrase_stemmed = []
        self.__tmp_phrase_pos = []
        self.__tmp_phrase_offset = []

    def add(self, raw: str, stemmed: str, pos: str, offset: int):
        """ add single token, integrate it as phrase or keep as part of phrase

         Parameter
        -------------------
        raw: token in raw form
        stemmed: stemmed token
        pos: Part of speech
        offset: offset position
        """

        def add_tmp_list():  # add to tmp list
            self.__tmp_phrase_raw.append(raw)
            self.__tmp_phrase_stemmed.append(stemmed)
            self.__tmp_phrase_pos.append(pos)
            self.__tmp_phrase_offset.append(offset)

        # phrase with more symbol than alphanumeric should be ignored
        if len(re.sub(r'\w', '', stemmed)) > len(re.sub(r'\W', '', stemmed)):
            pos = 'RANDOM'

        if pos == 'NOUN' or (pos == 'ADJ' and 'NOUN' not in self.__tmp_phrase_pos):
            add_tmp_list()
        else:
            if 'NOUN' in self.__tmp_phrase_pos:  # finalize list of tokens as phrase
                self.__add_to_structure()
            else:  # only adjective can't be regarded as phrase
                self.__initialize_list()

    def __add_to_structure(self):
        """ add to main dictionary and initialize tmp lists """
        phrase_stemmed = self.__joiner.join(self.__tmp_phrase_stemmed)
        # too many words or too long character should be ignored
        if (self.__maximum_word_number and len(self.__tmp_phrase_stemmed) > self.__maximum_word_number) or \
                (self.__maximum_char_number and len(phrase_stemmed) > self.__maximum_char_number):
            self.__initialize_list()
            return

        if len(self.__tmp_phrase_offset) == 1:
            offset = [self.__tmp_phrase_offset[0], self.__tmp_phrase_offset[0] + 1]
        else:
            offset = [self.__tmp_phrase_offset[0], self.__tmp_phrase_offset[-1]]

        # key of main dictionary
        if phrase_stemmed in self.__content.keys():
            self.__content[phrase_stemmed]['raw'] += [self.__joiner.join(self.__tmp_phrase_raw)]
            self.__content[phrase_stemmed]['offset'] += [offset]
            self.__content[phrase_stemmed]['count'] += 1
        else:
            self.__content[phrase_stemmed] = dict(
                stemmed=self.__joiner.join(self.__tmp_phrase_stemmed), pos=' '.join(self.__tmp_phrase_pos),
                raw=[self.__joiner.join(self.__tmp_phrase_raw)], offset=[offset], count=1)

        # initialize tmp lists
        self.__initialize_list()
        return


class PhraseConstructor:
    """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class """

    def __init__(self,
                 language: str = 'en',
                 stopwords_list: List = None,
                 maximum_word_number: int = 6,
                 maximum_char_number: int = 70):
        """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class

         Parameter
        ----------------------
        language: str
        stopwords_list: List
        """
        self.__language = language.lower()[:2]
        self.__stopwords = get_stopwords_list(self.__language, stopwords_list=stopwords_list)
        self.__maximum_word_number = maximum_word_number
        self.__maximum_char_number = maximum_char_number
        if self.__language == 'en':
            self.__stemmer = PorterStemmer()
            try:
                pos_tag(['this', 'is', 'a', 'sample'])
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            self.__pos_tagger = pos_tag
        else:
            raise ValueError('available only for English because of PoS tagger')

    @staticmethod
    def simplify_pos(_pos):
        if _pos in ["NN", "NNS", "NNP", "NNPS", "CD"]:
            return "NOUN"
        elif _pos in ["JJ", "JJR", "JJS"]:
            return "ADJ"
        return "RANDOM"

    def tokenize_and_stem(self, document: str, apply_stopwords: bool = True):
        _, stemmed, _ = self.preprocess(document)
        if apply_stopwords:
            return list(filter(lambda x: x.lower() not in self.__stopwords, stemmed))
        return stemmed

    def preprocess(self, document: str, flatten_sentence: bool = True):
        """ tokenization/stemming/PoS tagging """
        sentence_token = [[
            w for w in split_contractions(web_tokenizer(s)) if not (w.startswith("'") and len(w) > 1) and len(w) > 0
        ] for s in list(split_multi(document)) if len(s.strip()) > 0]
        stemmed = [list(map(lambda x: self.__stemmer.stem(x), words)) for words in sentence_token]
        pos = [list(map(lambda x: self.simplify_pos(x[1]), self.__pos_tagger(words))) for words in sentence_token]
        if flatten_sentence:
            return list(chain(*sentence_token)), list(chain(*stemmed)), list(chain(*pos))
        return sentence_token, stemmed, pos

    def tokenize_and_stem_and_phrase(self, document: str):
        """ tokenization & stemming & phrasing

         Parameter
        ---------------------
        document: str

         Return
        ---------------------
        `Phrase.phrase` object, stemmed_token
        """
        sentence_token, stemmed, pos = self.preprocess(document)

        # phraser instance
        phrase_structure = Phrase(self.__language == 'ja',
                                  maximum_word_number=self.__maximum_word_number,
                                  maximum_char_number=self.__maximum_char_number)
        n = 0
        for n, (t, s, p) in enumerate(zip(sentence_token, stemmed, pos)):
            if t.lower() in self.__stopwords or s.lower() in self.__stopwords:
                p = 'RANDOM'
            phrase_structure.add(raw=t, stemmed=s, pos=p, offset=n)

        # to finalize the phrase in the end of sentence
        phrase_structure.add(raw='.', stemmed='.', pos='PUNCT', offset=n + 1)

        return phrase_structure.phrase, stemmed,

    def force_reset_stopwords(self):
        self.__stopwords = []

    @property
    def stopwords(self):
        return self.__stopwords
