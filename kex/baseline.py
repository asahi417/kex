""" Implementation of heuristic methods """
from ._phrase_constructor import PhraseConstructor

__all__ = 'FirstN'


class FirstN:
    """ First N keywords extractor """

    def __init__(self, language: str = 'en', maximum_word_number: int = 3):
        """ First N keywords extractor """
        self.phrase_constructor = PhraseConstructor(language=language, maximum_word_number=maximum_word_number)
        self.prior_required = False

    def get_keywords(self, document: str, n_keywords: int = 10):
        """ Get keywords

         Parameter
        ------------------
        document: str
        n_keywords: int

         Return
        ------------------
        a list of dictionary consisting of 'stemmed', 'pos', 'raw', 'offset', 'count'.
        eg) {'stemmed': 'grid comput', 'pos': 'ADJ NOUN', 'raw': ['grid computing'], 'offset': [[11, 12]], 'count': 1}
        """
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        sorted_phrases = sorted(phrase_instance.values(), key=lambda x: x['offset'][0][0])
        sorted_phrases = sorted_phrases[:min(len(sorted_phrases), n_keywords)]
        for n, i in enumerate(sorted_phrases):
            sorted_phrases[n]['score'] = n

        return sorted_phrases
