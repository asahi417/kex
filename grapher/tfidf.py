""" Model to utilize TFIDF: Wrapper of gensim API for TFIDF """
import os
import logging
from logging.config import dictConfig

import gensim
import numpy as np
from gensim import corpora

from ._phrase_constructor import PhraseConstructor

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
CACHE_DIR = './cache/tfidf'
__all__ = 'TFIDF'


class TFIDF:
    """ TFIDF """

    def __init__(self, language: str = 'en'):
        self.__model = None
        self.__dict = None
        self.phrase_constructor = PhraseConstructor(language=language)

    def load(self, directory: str = None):
        """ load saved lda model and dictionary instance used to train the model """
        directory = CACHE_DIR if directory is None else directory
        path_to_model = os.path.join(directory, 'tfidf_model')
        path_to_dict = os.path.join(directory, 'tfidf_dict')

        assert os.path.exists(path_to_model), 'no model found: {}'.format(path_to_model)
        assert os.path.exists(path_to_dict), 'no dict found: {}'.format(path_to_dict)

        LOGGER.info('loading model...')
        self.__model = gensim.models.TfidfModel.load(path_to_model)
        self.__dict = gensim.corpora.Dictionary.load_from_text(path_to_dict)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])

    def save(self, directory: str):
        assert self.is_trained, 'training before run any inference'
        directory = CACHE_DIR if directory is None else directory
        os.makedirs(os.path.join(directory), exist_ok=True)
        self.__model.save(os.path.join(directory, 'tfidf_model'))
        self.__dict.save_as_text(os.path.join(directory, 'tfidf_dict'))

    def train(self, data: list, export_directory: str = None, normalize: bool = True):
        """ TFIDF training

         Parameter
        ----------------
        data: list of document (list of string)
        """
        # get stemmed token
        stemmed_tokens = [self.phrase_constructor.tokenization(d) for d in data]
        # build TFIDF
        LOGGER.info("building corpus...")
        self.__dict = corpora.Dictionary(stemmed_tokens)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])
        corpus = [self.__dict.doc2bow(text) for text in stemmed_tokens]
        LOGGER.info("training model...")
        self.__model = gensim.models.TfidfModel(corpus=corpus, normalize=normalize)
        LOGGER.info("saving model and corpus at {}".format(export_directory))
        self.save(export_directory)

    def distribution_word(self, document: str):
        """ word distribution of given document with pre-calculate TFIDF matrix

         Parameter
        ----------------
        tokens: document to get topic distribution

         Return
        -----------
        dict((word_0, prob_0), ..., (word_n, prob_n)) in order of probability
        Note that `n` is dynamically changing based on the coverage of probability and the probability itself will
        change slightly due to the randomness of sampling
        """
        assert self.is_trained, 'training before run any inference'
        # get stemmed token
        stemmed_tokens = self.phrase_constructor.tokenization(document)
        bow = self.__dict.doc2bow(stemmed_tokens)
        dist = dict((self.__dict.id2token[w_id], p) for w_id, p in self.__model[bow])
        return dist

    @property
    def model(self):
        """ gensim model instance of LDA """
        assert self.is_trained, 'training before run any inference'
        return self.__model

    @property
    def vocab_size(self):
        """ vocabulary size """
        assert self.is_trained, 'training before run any inference'
        return len(self.__dict.token2id)

    @property
    def vocab(self):
        """ vocabulary size """
        assert self.is_trained, 'training before run any inference'
        return list(self.__dict.token2id.keys())

    @property
    def dictionary(self):
        """ gensim dictionary instance """
        assert self.is_trained, 'training before run any inference'
        return self.__dict

    @property
    def is_trained(self):
        return self.__dict is not None and self.__model is not None

    def get_keywords(self, document: str, n_keywords: int = 10):
        """ Get keywords

         Parameter
        ------------------
        document: str
        n_keywords: int

         Return
        ------------------
        a list of keywords with score eg) [('aa', 0.5), ('b', 0.3), ..]
        """
        # convert phrase instance
        phrase_instance, stemmed_tokens = self.phrase_constructor.get_phrase(document)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
            return []

        dist_word = self.distribution_word(document)

        def aggregate_prob(__phrase_key):
            __phrase = phrase_instance[__phrase_key]
            prob = float(np.mean([[dist_word[_r] if _r in dist_word.keys() else 0 for _r in r.split()]
                                  for r in __phrase['raw']]))
            return prob

        phrase_prob = [(k, aggregate_prob(k)) for k in phrase_instance.keys()]

        # sorting
        phrase_score_sorted_list = sorted(phrase_prob, key=lambda id_score: id_score[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['raw'] = tmp['raw'][0]
            tmp['lemma'] = tmp['lemma'][0]
            tmp['n_source_tokens'] = len(stemmed_tokens)
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val
