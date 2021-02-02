""" LexSpec: Lexical Specificity based keyword extraction """
import os
import logging
import math
import json
from math import log, factorial, log10
from collections import Counter
from itertools import chain

from ._phrase_constructor import PhraseConstructor

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
CACHE_DIR = '{}/cache_kex/lexspec'.format(os.path.expanduser('~'))
__all__ = ('LexSpec', 'lexical_specificity', 'TF')


def average(_list):
    return sum(_list)/len(_list)


def lexical_specificity(T, t, f, k, lim: int = 400):
    """ calculate lexical specificity, originally implemented by Jose

    :param T: a size of reference corpus
    :param t: a size of sub-corpus (t <= T)
    :param f: a frequency of w in reference corpus
    :param k: a frequency of w in sub-corpus  (f <= k)
    :param lim:
    :return: float score
    """

    def stirling(n):
        return n * log(n) - n + (0.5 * log(2 * 3.141592 * n))

    assert t <= T, "t exceeds T"
    assert k <= f, "f ({}) exceeds k ({})".format(f, k)

    if f > lim:
        arg1 = stirling(f)
    else:
        arg1 = log(factorial(int(f)))
    if T - f > lim:
        arg2 = stirling(T - f)
    else:
        arg2 = log(factorial(int(T - f)))
    if t > lim:
        arg3 = stirling(t)
    else:
        arg3 = log(factorial(int(t)))
    if T - t > lim:
        arg4 = stirling(T - t)
    else:
        arg4 = log(factorial(int(T - t)))
    if T > lim:
        arg5 = stirling(T)
    else:
        arg5 = log(factorial(int(T)))
    if k > lim:
        arg6 = stirling(k)
    else:
        arg6 = log(factorial(int(k)))
    if f - k > lim:
        arg7 = stirling(f - k)
    else:
        arg7 = log(factorial(int(f - k)))
    if t - k > lim:
        arg8 = stirling(t - k)
    else:
        arg8 = log(factorial(int(t - k)))
    if T - f - t + k > lim:
        arg9 = stirling(T - f - t + k)
    else:
        arg9 = log(factorial(int(T - f - t + k)))

    prob = arg1 + arg2 + arg3 + arg4 - arg5 - arg6 - arg7 - arg8 - arg9
    first_prod = -log10(math.e) * prob

    if prob < log(0.1):
        prob1 = 1.0
        prob = 1.0
        while prob1 != 0.0 and (prob / prob1) < 10000000 and k <= f:
            prob2 = (prob1 * (f - k) * (t - k)) / ((k + 1) * (T - f - t + k + 1))
            prob = prob + prob2
            prob1 = prob2
            k += 1
        score = first_prod - log10(prob)
        return score
    else:
        return 0


class LexSpec:
    """ Lexical Specification based keyword extraction algorithm """

    def __init__(self, language: str = 'en'):
        self.__reference_corpus = None
        self.freq = None
        self.reference_corpus_size = 0
        self.prior_required = True
        self.phrase_constructor = PhraseConstructor(language=language)

    def load(self, directory: str = CACHE_DIR):
        """ load saved lda model and dictionary instance used to train the model """
        path_to_dict = "{}/lexical_specificity_frequency.json".format(directory)
        assert os.path.exists(path_to_dict), 'no dictionary found: {}'.format(path_to_dict)
        with open(path_to_dict, 'r') as f:
            self.freq = json.load(f)
        self.reference_corpus_size = sum(self.freq.values())
        logging.debug('loaded frequency dictionary from {}'.format(path_to_dict))

    def train(self, data: list, export_directory: str = CACHE_DIR):
        """ cache a dictionary {key: a word, value: occcurence of the word}

         Parameter
        ----------------
        data: a list of document (list of string)
        """
        # get stemmed token
        stemmed_tokens = list(chain(*map(lambda x: self.phrase_constructor.tokenize_and_stem(x), data)))
        self.freq = dict(Counter(stemmed_tokens))
        self.reference_corpus_size = sum(self.freq.values())
        self.save(export_directory)
        logging.debug('compute frequency dictionary saved at {}'.format(export_directory))

    def save(self, directory: str = './cache/priors/lexical_specificity'):
        assert self.is_trained, 'training before run any inference'
        os.makedirs(os.path.join(directory), exist_ok=True)
        with open("{}/lexical_specificity_frequency.json".format(directory), "w") as f:
            json.dump(self.freq, f)

    @property
    def is_trained(self):
        return self.freq is not None

    def lexical_specificity(self, document: str):
        stemmed_tokens = self.phrase_constructor.tokenize_and_stem(document)
        sub_freq = dict(Counter(stemmed_tokens))

        # compute lexical specificity
        sub_corpus_size = sum(sub_freq.values())
        ls_dict = {k: lexical_specificity(
            T=self.reference_corpus_size,
            t=sub_corpus_size,
            f=self.freq[k],
            k=sub_freq[k]
        ) for k in sub_freq.keys() if k in self.freq.keys()}
        return ls_dict

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
        assert self.is_trained, 'provide prior before running inference'

        # convert phrase instance
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
            return []

        sub_freq = dict(Counter(stemmed_tokens))

        # compute lexical specificity
        sub_corpus_size = sum(sub_freq.values())
        ls_dict = {k: lexical_specificity(
            T=self.reference_corpus_size,
            t=sub_corpus_size,
            f=self.freq[k],
            k=sub_freq[k]
        ) for k in sub_freq.keys() if k in self.freq.keys()}

        # aggregate score over individual word in a phrase
        def aggregate_prob(__phrase_key):
            __phrase = phrase_instance[__phrase_key]
            prob = float(average([
                ls_dict[_r] if _r in ls_dict.keys() else 0 for _r in __phrase['stemmed'].split()]))
            return prob

        phrase_prob = [(k, aggregate_prob(k)) for k in phrase_instance.keys()]

        # sorting
        phrase_score_sorted_list = sorted(phrase_prob, key=lambda id_score: id_score[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['n_source_tokens'] = len(stemmed_tokens)
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val


class TF(LexSpec):
    """ Term frequency based keyword extraction algorithm """

    def __init__(self, *args, **kwargs):
        """ Term frequency based keyword extraction algorithm """
        super(TF, self).__init__(*args, **kwargs)

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
        assert self.is_trained, 'provide prior before running inference'

        # convert phrase instance
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
            return []

        sub_freq = dict(Counter(stemmed_tokens))

        # compute term frequency
        tf_dict = {k: self.freq[k] for k in sub_freq.keys() if k in self.freq.keys()}

        # aggregate score over individual word in a phrase
        def aggregate_prob(__phrase_key):
            __phrase = phrase_instance[__phrase_key]
            prob = float(average([
                tf_dict[_r] if _r in tf_dict.keys() else 0 for _r in __phrase['stemmed'].split()]))
            return prob

        phrase_prob = [(k, aggregate_prob(k)) for k in phrase_instance.keys()]

        # sorting
        phrase_score_sorted_list = sorted(phrase_prob, key=lambda id_score: id_score[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['n_source_tokens'] = len(stemmed_tokens)
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val

