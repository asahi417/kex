import unicodedata
import re
import os
from .stop_word import StopwordDetection, SkipPhrase

DEV_MODE = os.getenv('DEV_MODE', '')  # if `without_stopword`, skip stopword detection


def only_roman_chars(unistr):
    latin_letters = {}

    def is_latin(uchr):
        try:
            return latin_letters[uchr]
        except KeyError:
            return latin_letters.setdefault(uchr, 'LATIN' in unicodedata.name(uchr))

    return all(is_latin(_uchr) for _uchr in unistr if _uchr.isalpha())


# Minimum character length of keyphrase (less than this values will be eliminated from candidate lists)
MIN_LENGTH = 2


def exclude_ticker(string):
    """ Filter to exclude ticker and some tag, indicating news

    - news tag
    [ID:~]
    [nL4N1V7386]

    - ticker, which has various format
    [IGO.AX]
    <TEF.MC>,
    <.STOXX>
    <0#.INDEX>
    [O/SING1]
    <MKS.L>.
    """

    # convert html tag
    string = string.replace('&amp;', '&')
    string = string.replace('&lt;', '<')
    string = string.replace('&gt;', '>')
    # remove ticker
    string = re.sub(r'<[^<>]*>', '', string)
    string = re.sub(r'\[[^[\]]*]', '', string)
    return string


class CleaningStr:
    """ Cleaning strings by regx contains

     List of feature
    --------------------
    * killed ticker for english (see above `exclude_ticker`)
    * remove redundant half-space (make them single half-space)
        * more than 2 repetition of half-space will be replaced by `.`
    * convert full width to half width
    * unify web url as ID.URL, numeric term as ID.NUMBER, and mail address as ID.MAIL
    """

    def __init__(self,
                 language: str = 'en',
                 skip_phrase: list = None):
        self.__language = language
        self.__skip_phrase_detector = SkipPhrase(skip_phrase)

    def process(self, sentence: str):
        if self.__language == 'en':
            sentence = exclude_ticker(sentence)
            if not only_roman_chars(sentence):
                # avoid to get non-roman, like jp, russian, arabic
                return ''
        if self.__skip_phrase_detector.apply(sentence):
            return ''
        sentence = self.single_halfspace(sentence, split=2)
        sentence = self.full_to_half(sentence)
        sentence = self.web_url(sentence, self.__language)
        sentence = self.mail_address(sentence, self.__language)
        sentence = self.numeric(sentence, self.__language)
        sentence = self.single_halfspace(sentence, split=10)
        sentence = self.convert_identifier(sentence)
        return sentence

    def convert_identifier(self, sentence):
        for k, v in self.identifier.items():
            sentence = sentence.replace(k, v)
        return sentence

    @property
    def identifier(self):
        return {'ID.URL': 'XXX',
                'ID.NUMBER': 'YYY',
                'ID.MAIL': 'ZZZ'}

    @staticmethod
    def full_to_half(sentence: str):
        """ full-width to half-width

        - original
        ['ａａａ']
        - converted
        ['aaa']
        """
        return unicodedata.normalize("NFKC", sentence)

    @staticmethod
    def web_url(sentence: str,
                language: str = 'en'):
        """ convert URL

        - original
        ['http://soundcloud.com I have www.facebook.com']
        - converted
        ['ID.URL I have ID.URL']
        """
        if language == 'en':
            sentence = re.sub(r'\b(https?://[\w./]*|www[.][\w./]*)\b', r'ID.URL', sentence)
            sentence = re.sub(r'[^\s]*ID.URL[^\s]*', r'ID.URL', sentence)
        elif language == 'jp':
            sentence = re.sub(r'[\s]*https?://[\w./]*|www[.][\w./]*[\s]*', r' ID.URL ', sentence)
            # sentence = re.sub(r'[^\s]*ID.URL[^\s]*', r'ID.URL', sentence)
        return sentence

    @staticmethod
    def mail_address(sentence: str,
                     language: str = 'en'):
        """ convert mail address

        - original
        ['Here is my e-mail aushio@cogent.co.jp']
        - converted
        ['Here is my e-mail ID.MAIL']
        """
        if language == 'en':
            sentence = re.sub(r'\b[^\s]+@[\w.]+([^\s])\b', r'ID.MAIL', sentence)
            sentence = re.sub(r'[^\s]*(ID.MAIL)[^\s]*', r'ID.MAIL', sentence)
        elif language == 'jp':
            sentence = re.sub(r'[\s]*[^\s]+@[\w.]+([^\s])[\s]*', r' ID.MAIL ', sentence)
        return sentence

    @staticmethod
    def single_halfspace(sentence: str,
                         split: int = 3):
        """ replace multiple half-space to single one, if half space repeating more than split `split`,
        add `.`

        - original
        ['  wow  wow wow  wow  ']
        - converted
        ['wow wow wow wow']
        """

        sentence = re.sub(r'([^\.^\s])(\s){%i,100000}' % split, r'\1 . ', sentence)
        sentence = re.sub(r"(\s+)(\S+)(\s+)", r" \2 ", " " + sentence + " ")[1:-1]
        sentence = re.sub(r"\s+\Z", r"", sentence)
        return sentence

    @staticmethod
    def numeric(sentence: str,
                language: str = 'en'):
        """ convert numeric term, except for named entity (eg, G5, AKB48)

        - original
        ['I have attended 34th ICML. 090 8899 5343', 'G5', '<0027.HK>', '[ nL8N1C350B ]', 'AKB48']
        - converted
        ['I have attended ID.NUMBER ICML. ID.NUMBER ID.NUMBER ID.NUMBER', 'G5', 'ID.NUMBER', '[ ID.NUMBER ]', 'AKB48']
        """
        if language == 'en':
            sentence = re.sub(r'\b[^\s^A-Z]+[^\s^\d]*\d+[^\s]*\b', r'ID.NUMBER', sentence)
            sentence = re.sub(r'[^\s]*ID.NUMBER[^\s]*', r'ID.NUMBER', sentence)
        return sentence


class CandidatePhrase:
    """ Data structure for candidate phrasing, which basically follow regx bellow
    [Adjective]*[Noun|proper noun]+

    The structure contains a main dictionary, in which key is the stemmed candidates (so, phrases which share same
    stemmed form are regarded as same phrase)
        * pos: pos
        * stemmed: stemmed form
        * raw: list of raw phrase
        * lemma: list of lemma
        * offset: offset
    Also exclude token in given stop word list.
    """

    def __init__(self,
                 join_without_space: bool=False,
                 add_verb: bool = False):

        self.__content = dict()
        self.__initialize_list()
        self.__join_without_space = join_without_space
        self.__stopword_detector = StopwordDetection()
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

    def __stopwords(self, string: str, phrase: bool = False):
        """ check if string is in stopwords"""
        # length check
        if len(string) < MIN_LENGTH:
            return True
        else:
            if DEV_MODE == 'without_stopword':
                return False
            else:
                if phrase:
                    return self.__stopword_detector.apply_phrase(string)
                else:
                    return self.__stopword_detector.apply(string)

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

        if pos in ['ADJ'] and not self.__stopwords(raw):
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase if it's not in the stop phrase
                self.__add_to_structure()
            add_tmp_list()

        elif self.__add_verb and pos in ['VERB'] and not self.__stopwords(raw):
            if 'NOUN' in self.__tmp_phrase_pos or 'PROPN' in self.__tmp_phrase_pos:
                # finalize list of tokens as phrase if it's not in the stop phrase
                self.__add_to_structure()
            add_tmp_list()
            # single verb will be a phrase
            self.__add_to_structure()

        elif pos in ['NOUN', 'PROPN'] and not self.__stopwords(raw):
            add_tmp_list()
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
        _raw = _join.join(self.__tmp_phrase_raw)
        if self.__stopwords(_raw, phrase=True):
            self.__initialize_list()
            return

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
                count=1
            )
        # initialize tmp lists
        self.__initialize_list()
        return
