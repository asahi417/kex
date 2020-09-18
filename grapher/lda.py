""" Model to utilize Topic Model: Wrapper of gensim API for Topic Model with LDA """
import os
import logging
from logging.config import dictConfig

import gensim
from gensim import corpora

from ._phrase_constructor import PhraseConstructor

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
CACHE_DIR = './cache/lda'
__all__ = 'LDA'


class LDA:
    """ LDA """

    def __init__(self, language: str = 'en'):
        self.__model = None
        self.__dict = None
        self.phrase_constructor = PhraseConstructor(language=language)

    def load(self, directory: str = None):
        """ load saved lda model and dictionary instance used to train the model """
        directory = CACHE_DIR if directory is None else directory
        path_to_model = os.path.join(directory, 'lda_model')
        path_to_dict = os.path.join(directory, 'lda_dict')

        assert os.path.exists(path_to_model), 'no model found: {}'.format(path_to_model)
        assert os.path.exists(path_to_dict), 'no dict found: {}'.format(path_to_dict)

        LOGGER.info('loading model...')
        self.__model = gensim.models.ldamodel.LdaModel.load(path_to_model)
        self.__dict = gensim.corpora.Dictionary.load_from_text(path_to_dict)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])

    def save(self, directory: str):
        assert self.is_trained, 'training before run any inference'
        directory = CACHE_DIR if directory is None else directory
        os.makedirs(os.path.join(directory), exist_ok=True)
        self.__model.save(os.path.join(directory, 'lda_model'))
        self.__dict.save_as_text(os.path.join(directory, 'lda_dict'))

    def train(self, data: list, export_directory: str = None, num_topics: int = 15):
        """ LDA training

         Parameter
        ----------------
        data: list of document (list of string)
        num_topics: number of topic
        """
        stemmed_tokens = [self.phrase_constructor.tokenize_and_stem(d) for d in data]
        # build LDA model
        LOGGER.info("building corpus...")
        self.__dict = corpora.Dictionary(stemmed_tokens)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])
        corpus = [self.__dict.doc2bow(text) for text in stemmed_tokens]
        LOGGER.info("training model...")
        self.__model = gensim.models.ldamodel.LdaModel(
            corpus=corpus, num_topics=num_topics, id2word=self.__dict)
        LOGGER.info("saving model and corpus at {}".format(export_directory))
        self.save(export_directory)

    def distribution_topic_document(self, document: str):
        """ topic probability distribution of given documents p(z|d = tokens)

         Parameter
        ----------------
        tokens: document to get topic distribution

         Return
        -----------
        [(topic_id, prob), (topic_id, prob), ...] in order of prob
        """
        assert self.is_trained, 'training before run any inference'
        stemmed_tokens = self.phrase_constructor.tokenize_and_stem(document)
        bow = self.__dict.doc2bow(stemmed_tokens)
        topic_dist = self.__model.get_document_topics(bow, minimum_probability=0.0)
        return topic_dist

    def probability_word(self, tokens: list = None):
        """ individual probability of given tokens conditioned by topic: [p(w_i|z) for w_i in tokens]

         Parameter
        ----------------
        tokens: list of token, each of them is treated as independently and calculate probability conditioned
                by each topic

         Return
        -----------
        ndarray (topic_number, len(tokens)): element (j, i) corresponds to a probability of token_i conditioned by
        topic j, p(w_i|z_j). If tokens is not provided, return entire probability distribution.
        """
        assert self.is_trained, 'training before run any inference'
        topic_table = self.__model.get_topics()
        if tokens is None:
            return topic_table
        else:
            ids = [self.__dict.token2id[t] for t in tokens]
            topic_dist = topic_table[:, ids]
            return topic_dist

    @property
    def is_trained(self):
        return self.__dict is not None and self.__model is not None

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
    def topic_size(self):
        """ topic size """
        assert self.is_trained, 'training before run any inference'
        return self.__model.num_topics

    @property
    def dictionary(self):
        """ gensim dictionary instance """
        assert self.is_trained, 'training before run any inference'
        return self.__dict
