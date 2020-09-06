""" Stopwords
- StopwordDetection
    This is aimed to decide if a given single phrase is stopword or not. Note that, this is for single phrase
    detection. Don't directly apply to whole sentence.

    >>> detector = StopwordDetection()
    >>> detector.apply('Apple')
        False
    >>> detector.apply('next week')
        True

    You can add new stopword by specify as initial parameter
    >>> detector = StopwordDetection(['Apple'])
    >>> detector.apply('Apple')
        True

- Skip Phrase
    This is aimed to remove junk documents by string inclusion of certain phrase, which only occurred in junk data,
    which is in many case a table only document, or summary of various kind of data.

    >>> detector = SkipPhrase()
    >>> detector.apply('This is not a junk document')
        False
    >>> detector.apply('The phrase `Daily Margin Long Amount` implies the document is shit')
        True

    You can add new phrase by specify as initial parameter
    >>> detector = SkipPhrase(['This is not a junk document'])
    >>> detector.apply('This is not a junk document')
        True

"""


import re
import string
from nltk.corpus import stopwords


class StopwordDetection:
    """ Stopword collection

    - stopwords will be applied to single word eg) date, year, man
    - stopphrase will be applied to single phrase (multiple words) eg) Breaking news, Top stories
    """

    # exceptions to be ignored from stopwords detection (possible to be people name)
    exclude = ['May', 'Jun']

    # identifier from `basic_modules.CleaningStr`
    stopwords_cleaner_id = ['XXX', 'YYY', 'ZZZ']

    # generic stopwords list (english)
    try:
        stopwords_nltk = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download('stopwords')
        stopwords_nltk = stopwords.words("english")

    # generic stopwords list (japanese)
    stopwords_common_jp = [
        "する", "なる", "れる", "まくる", "てる", "てる", "よう", "ある", "しろ", "くる", "みる", "たち", "ここ", "ない",
        "とり", "れろ", "いる", "られる", "くれる", "こと", "これ", "あれ", "それ", "そう", "できる", "氏", "さん", "的",
        "ため", "いう", "はず", 'ぶり', 'つま', '我々', '私', '僕', 'あなた', '自分', '前日', 'あと', '大事', 'ここに',
        'テスト'
    ]

    # from reuters data
    stopwords_reuters_news = ['’s'] + list(string.ascii_lowercase)
    stopphrase_reuters_news = [
        'amp', 'pct', 'yr', 'yr/yr', 'q1', 'q2', 'q3', 'q4', '-sources', 'co.', 'co', 'inc.', 'inc', 'percent',
        'gmt', 'note', 'poll', 'polls', 'quarter', 'double click', 'click', 'amount', 'qtr', 'atm',
        'details', 'value', 'keywords', 'keyword', 'part', 'author', 'comments',
        'country', 'prices', 'reporters', 'data', 'polls', 'index', 'FED/DIARY', 'M/DIARY', 'Editing',
        'yesterday', 'tomorrow', 'today',
        'TIMES EST/GMT', 'top stories', 'additional details', 'type', 'typo',
        'bln', 'bn', 'b', 'mln', 'ml', 'm', 'k'
    ]

    # custom regx filter from reuters data
    stopwords_custom_regx_reuters_news = [
        # r'\A[^(gt)^(lt)]*(lt|gt)[;]*[\S]*\Z',  # html tag (\gt, \lt)
        # r'\A(\S)*(&amp)[;]*(\S)*\Z',
        # r'\A([xX])+\S*\Z',
        # r'\A[^a-zA-Z]+',  # begin with non-alphabet
        r'\A[\W]+',  # begin with non-alphanumeric
        r'\A(ID)+.\S*',  # start from `ID.`
        r'\A(breakingview)[s]*'  # phrase including `breakingview[s]`
    ]

    stopphrase_custom_regx_reuters_news = [
        r'\A[\d.]+\S*\Z',  # begin with numeric and single word
        # r'\A([\d.]+\S*)+\Z',  # begin with numeric and single word
        r'(�)+',  # remove unknown character
        r'[.-]{3}'  # remove i
    ]

    # about date: anything related to numeric term
    date = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
        'December',
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays',
        'Mon', 'Tues', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',
        'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
        'sec', 'min', 'wk', 'mth', 'yr', 'yy',
        'yr.yr', 'yr-to-date', 'yy-to-date', 'mth-to-date', 'wk-to-date',
        'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years',
        'bln', 'bn', 'b', 'mln', 'ml', 'm', 'k', 'time', 'low', 'floor', 'total', 'atm'
    ]
    date_adjectives = ['next', 'last', 'best', 'latest', 'mid', 'yy']
    date_subjectives = []
    date_prefix = ['\-', 'mid\-', 'yr\-end']  # noqa TODO
    date_suffix = [',', '\.']  # noqa TODO

    @staticmethod
    def list_to_regx(list_of_string):
        _stopwords = '|'.join(list_of_string).lower()
        _stopwords_regx = r'\A\s*(%s)\s*\Z' % _stopwords
        return _stopwords_regx

    def get_rule_date(self):
        suffix = ''.join(['(%s)' % s.lower() for s in self.date_suffix])
        prefix = ''.join(['(%s)' % s.lower() for s in self.date_prefix])
        adjectives = '|'.join(self.date_adjectives).lower()
        subjectives = '|'.join(self.date_subjectives).lower()
        date = '|'.join(self.date).lower()

        suffix_regx = r'[%s]*' % suffix
        prefix_regx = r'[%s]*' % prefix
        adjectives_regx = r'(%s)*' % adjectives
        subjectives_regx = r'(%s)*' % subjectives
        date_regx = r'(%s)' % date
        numeric_regx = r'[\d.]+\S*'

        final_regx_list = [
            r'\A(\s*%s\s*%s\b%s\b%s\s*%s\s*)+\Z'
            % (adjectives_regx, prefix_regx, date_regx, suffix_regx, subjectives_regx),
            r'\A(\s*%s\s*%s%s%s\s*%s\s*)+\Z'
            % (numeric_regx, prefix_regx, date_regx, suffix_regx, subjectives_regx)
        ]
        return final_regx_list

    def __init__(self):

        ################################
        # single token based stopwords #
        ################################
        # get token-based filter
        self.regx_full_list_token = [
            self.list_to_regx(self.stopwords_cleaner_id),
            self.list_to_regx(self.stopwords_reuters_news),
            self.list_to_regx(self.stopwords_nltk),
            self.list_to_regx(self.stopwords_common_jp)
        ] + self.stopwords_custom_regx_reuters_news
        ##########################
        # phrase based stopwords #
        ##########################
        # regx full list
        self.regx_full_list_phrase = [
            self.list_to_regx(self.stopphrase_reuters_news)
        ]
        self.regx_full_list_phrase += self.stopphrase_custom_regx_reuters_news
        self.regx_full_list_phrase += self.get_rule_date()

    def apply(self,
              _string: str,
              debug: bool = False):

        if _string in self.exclude:
            return False

        for _rule in self.regx_full_list_token:
            if re.search(_rule, _string.lower()):
                if debug:
                    print(_rule)
                return True
        return False

    def apply_phrase(self,
                     _string: str,
                     debug: bool = False):

        if _string in self.exclude:
            return False

        for _rule in self.regx_full_list_phrase:
            if re.search(_rule, _string.lower()):
                if debug:
                    print(_rule)
                return True
        return False


class SkipPhrase:
    """ Phrase to skip: avoid junk documents """

    phrase_table = [
        "today's Margin Long Amount",
        "Daily Margin Long Amount",
        "reported the following daily data",
        "following are daily statistics",
        "For all coming Fed events",
        "Reuters produces dozens of polls of professionals every month that provide exclusive insight into economies and markets",  # noqa TODO
        "Near-term pipeline of notable IPOs"
    ]

    @staticmethod
    def list_to_regx(list_of_string):
        _stopwords = '|'.join(list_of_string)
        _stopwords_regx = r'\s*(%s)\s*' % _stopwords
        return _stopwords_regx

    def __init__(self,
                 new_skip_phrase: list = None):
        self.__reg_phrase = [
            self.list_to_regx(self.phrase_table)
        ]
        if new_skip_phrase is not None:
            self.__reg_phrase += [self.list_to_regx(new_skip_phrase)]

    def apply(self,
              _string,
              debug: bool = False):
        for _rule in self.__reg_phrase:
            if re.search(_rule, _string):
                if debug:
                    print(_rule)
                return True
        return False
