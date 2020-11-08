import re
from typing import List

import nltk
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex
from nltk import SnowballStemmer
from nltk.corpus import stopwords

from ._tokenizer_ja import TokenizerJa

nltk.download('stopwords')

__all__ = 'PhraseConstructor'
escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '-lcb-': '{', '-rcb-': '}'}


class Phrase:
    """ Data structure for phrases, which follows regx [Adjective]*[Noun|proper noun]+

    The structure contains a main dictionary, in which key is the stemmed candidates (so, phrases which share same
    stemmed form are regarded as same phrase)
        * pos: pos
        * stemmed: stemmed form
        * raw: list of raw phrase
        * lemma: list of lemma
        * offset: offset
    Also exclude token in given stop word list.
    """

    def __init__(self, join_without_space: bool = False, stopword: List = None):
        """ Data structure for phrases, which follows regx [Adjective]*[Noun|proper noun]+

         Parameter
        -------------------
        join_without_space: bool
            join token without halfspace to construct a phrase (eg, Japanese)
        add_verb: bool
            add verb as a part of phrase
        """

        self.__content = dict()
        self.__initialize_list()
        self.__join_without_space = join_without_space
        self.__stopword = [] if stopword is None else stopword

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

        def add_tmp_list():
            # add to tmp list
            self.__tmp_phrase_raw.append(raw)
            self.__tmp_phrase_stemmed.append(stemmed)
            self.__tmp_phrase_pos.append(pos)
            self.__tmp_phrase_offset.append(offset)

        if raw.lower() in self.__stopword or stemmed.lower() in self.__stopword:
            pass
        elif pos in ['ADJ']:
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase if it's not in the stop phrase
                self.__add_to_structure()
            add_tmp_list()
            return
        elif pos in ['NOUN', 'PROPN']:
            add_tmp_list()
            return

        if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
            # finalize list of tokens as phrase
            self.__add_to_structure()
        else:
            # only adjective can't be regarded as phrase
            self.__initialize_list()
        return

    def __add_to_structure(self):
        """ add to main dictionary and initialize tmp lists """
        _join = '' if self.__join_without_space else ' '
        if len(self.__tmp_phrase_offset) == 1:
            offset = [self.__tmp_phrase_offset[0], self.__tmp_phrase_offset[0] + 1]
        else:
            offset = [self.__tmp_phrase_offset[0], self.__tmp_phrase_offset[-1]]
        # key of main dictionary
        phrase_stemmed = _join.join(self.__tmp_phrase_stemmed)
        if phrase_stemmed in self.__content.keys():
            self.__content[phrase_stemmed]['raw'] += [_join.join(self.__tmp_phrase_raw)]
            self.__content[phrase_stemmed]['offset'] += [offset]
            self.__content[phrase_stemmed]['count'] += 1
        else:
            self.__content[phrase_stemmed] = dict(
                stemmed=_join.join(self.__tmp_phrase_stemmed),
                pos=' '.join(self.__tmp_phrase_pos),
                raw=[_join.join(self.__tmp_phrase_raw)],
                offset=[offset],
                count=1)

        # initialize tmp lists
        self.__initialize_list()
        return


class PhraseConstructor:
    """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class """

    def __init__(self, language: str = 'en'):
        """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class

         Parameter
        ----------------------
        language: str
            tokenization depends on Language
        no_stemming: bool
            no stemming is applied (True if the document is already stemmed)
        """
        # setup spacy nlp model
        self.__language = language
        if self.__language == 'en':
            self.__spacy_processor = self.setup_spacy_processor()
            self.__stemming = SnowballStemmer('english')
            self.__tokenizer_ja = None
            self.__stop_words = stopwords.words('english')
        elif self.__language == 'ja':
            self.__spacy_processor = None
            self.__stemming = None
            self.__stop_words = None
            self.__tokenizer_ja = TokenizerJa()
        else:
            raise ValueError('undefined language (only en/ja are supported): {}'.format(language))

    def preprocess(self, string: str):
        if self.__language == 'en':
            if string.lower() in escaped_punctuation.keys():
                string = escaped_punctuation[string.lower()]
        return string

    @staticmethod
    def setup_spacy_processor():
        """ setup spacy tokenizer """
        try:
            spacy_processor = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            spacy_processor = spacy.load('en_core_web_sm')

        def custom_tokenizer():
            # avoid splitting by `-`: ex) `off-site apple store` -> ['off-site', 'apple', 'store']
            # avoid splitting by `.`: ex) `U.S. and U.K.` -> ['U.S.', 'and', 'U.K.']
            # infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
            infix_re = re.compile(r'''[\,\?\:\;\‘\’\`\“\”\"\'~]''')
            prefix_re = compile_prefix_regex(spacy_processor.Defaults.prefixes)
            suffix_re = compile_suffix_regex(spacy_processor.Defaults.suffixes)

            return Tokenizer(spacy_processor.vocab,
                             prefix_search=prefix_re.search,
                             suffix_search=suffix_re.search,
                             infix_finditer=infix_re.finditer,
                             token_match=None)

        spacy_processor.tokenizer = custom_tokenizer()
        return spacy_processor

    def tokenize_and_stem(self, document: str):
        """ tokenization & stemming """
        if self.__language == 'en':
            return [self.__stemming.stem(self.preprocess(spacy_object.text))
                    for spacy_object in self.__spacy_processor(document)]
        elif self.__language == 'ja':
            return [lemma for pos, lemma, raw in self.__tokenizer_ja.tokenize(document)]
        else:
            raise ValueError('undefined language: {}'.format(self.__language))

    def tokenize_and_stem_and_phrase(self, document: str):
        """ tokenization & stemming & phrasing

         Parameter
        ---------------------
        document: str

         Return
        ---------------------
        `Phrase.phrase` object, stemmed_token
        """
        phrase_structure = Phrase(join_without_space=self.__language in ['ja'], stopword=self.__stop_words)
        stemmed_tokens = []
        n = 0
        if self.__language == 'en':
            for n, spacy_object in enumerate(self.__spacy_processor(document)):
                raw = self.preprocess(spacy_object.text)
                stem = self.__stemming.stem(raw)
                stemmed_tokens.append(stem)
                phrase_structure.add(raw=raw, stemmed=stem, pos=spacy_object.pos_, offset=n)
        elif self.__language == 'ja':
            for n, (_pos, _lemma, _raw) in enumerate(self.__tokenizer_ja.tokenize(document)):
                stemmed_tokens.append(_lemma)
                phrase_structure.add(raw=_raw, stemmed=_lemma, pos=_pos, offset=n)
        else:
            raise ValueError('undefined language: {}'.format(self.__language))
        # to finalize the phrase in the end of sentence
        phrase_structure.add(raw='.', stemmed='.', pos='PUNCT', offset=n + 1)
        return phrase_structure.phrase, stemmed_tokens
