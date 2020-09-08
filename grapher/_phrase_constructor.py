import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex
from nltk import SnowballStemmer
import re

__all__ = 'PhraseConstructor'


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

    def __init__(self, join_without_space: bool = False, add_verb: bool = False):
        """ Data structure for phrases, which follows regx [Adjective]*[Noun|proper noun]+

        :param join_without_space: join token without halfspace to construct a phrase (eg, Japanese)
        :param add_verb: add verb as a part of phrase
        """

        self.__content = dict()
        self.__initialize_list()
        self.__join_without_space = join_without_space
        self.__add_verb = add_verb

    @property
    def phrase(self):
        return self.__content

    def __initialize_list(self):
        self.__tmp_phrase_raw = []
        self.__tmp_phrase_lemma = []
        self.__tmp_phrase_stemmed = []
        self.__tmp_phrase_pos = []
        self.__tmp_phrase_offset = []

    def add(self,
            raw: str,
            lemma: str,
            stemmed: str,
            pos: str,
            offset: int):
        """ add single token, integrate it as phrase or keep as part of phrase

        :param raw:
        :param lemma:
        :param stemmed:
        :param pos:
        :param offset: offset position
        """

        def add_tmp_list():
            # add to tmp list
            self.__tmp_phrase_raw.append(raw)
            self.__tmp_phrase_lemma.append(lemma)
            self.__tmp_phrase_stemmed.append(stemmed)
            self.__tmp_phrase_pos.append(pos)
            self.__tmp_phrase_offset.append(offset)

        if pos in ['ADJ']:
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase if it's not in the stop phrase
                self.__add_to_structure()
            add_tmp_list()
        elif self.__add_verb and pos in ['VERB']:
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase if it's not in the stop phrase
                self.__add_to_structure()
            add_tmp_list()
            # single verb will be a phrase
            self.__add_to_structure()
        elif pos in ['NOUN', 'PROPN']:
            add_tmp_list()
        # elif raw in SKIP_SYMBOL:
        #     add_tmp_list()
        else:
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase
                self.__add_to_structure()
            else:
                # only adjective can't be regarded as phrase
                self.__initialize_list()

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
            self.__content[phrase_stemmed]['lemma'] += [_join.join(self.__tmp_phrase_lemma)]
            self.__content[phrase_stemmed]['offset'] += [offset]
            self.__content[phrase_stemmed]['count'] += 1
        else:
            self.__content[phrase_stemmed] = dict(
                stemmed=_join.join(self.__tmp_phrase_stemmed),
                pos=' '.join(self.__tmp_phrase_pos),
                raw=[_join.join(self.__tmp_phrase_raw)],
                lemma=[_join.join(self.__tmp_phrase_lemma)],
                offset=[offset],
                count=1)

        # initialize tmp lists
        self.__initialize_list()
        return


class PhraseConstructor:
    """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class """

    def __init__(self, language: str = 'en', add_verb: bool = False):
        """ Phrase constructor to extract a list of phrase from given sentence based on `Phrase` class

        :param add_verb: add verb as a part of phrase
        :param language: tokenization depends on Language
        """
        # setup spacy nlp model
        self.__language = language
        if self.__language == 'en':
            self.__spacy_processor = self.setup_spacy_processor()
            self.__stemming = SnowballStemmer('english')
        else:
            raise ValueError('undefined language: {}'.format(language))
        self.__add_verb = add_verb

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

    def tokenization(self, document: str):
        """ tokenization """
        return [spacy_object.text for spacy_object in self.__spacy_processor(document)]

    def get_phrase(self, document: str):
        """ get phrase

        :param document:
        :return: `Phrase.phrase` object, tokens
        """
        phrase_structure = Phrase(add_verb=self.__add_verb)
        tokens = []
        n = 0
        for n, spacy_object in enumerate(self.__spacy_processor(document)):
            tokens.append(spacy_object.text)
            phrase_structure.add(
                raw=spacy_object.text, lemma=spacy_object.lemma_, stemmed=self.__stemming.stem(spacy_object.text),
                pos=spacy_object.pos_, offset=n)

        # to finalize the phrase in the end of sentence
        phrase_structure.add(raw='.', lemma='.', stemmed='.', pos='PUNCT', offset=n+1)
        return phrase_structure.phrase, tokens
