""" Model to utilize Topic Model: Wrapper of gensim API for Topic Model with LDA

WARNING: only for English!!!
"""

import os
import gensim
from gensim import corpora
from .processing import Processing
from ..util import create_log

DEFAULT_ROOD_DIR = os.path.join(os.path.expanduser("~"), 'keyphraser_data')


class LDA:
    """ LDA for Topic Model

     Usage
    ------------------
    >>> docs = [
    'doc1\nmore\nmore',
    'doc2\nmore\nmore',
    'doc3\nmore\nmore']
    >>> lda = LDA()
    # train model
    >>> lda.train(docs, 3, path_to_model='test_model', path_to_dict='test_dict')
    # or load model
    >>> lda.load(path_to_model='test_model', path_to_dict='test_dict')
    >>> lda.distribution_word_topic(0)
    >>> lda.probability_word(['graph'])

    """

    def __init__(self,
                 root_dir: str = None,
                 debug: bool = False,
                 language: str = 'en'):
        self.__lan = language
        self.__model = None
        self.__dict = None
        self.__logger = create_log() if debug else None
        if root_dir is None:
            self.__root_dir = os.path.join(DEFAULT_ROOD_DIR, 'lda')
        else:
            self.__root_dir = os.path.join(root_dir, 'lda')

    def load(self, data_name: str):
        """ load saved lda model and dictionary instance used to train the model

         Parameter
        ----------------
        path_to_model: (optional) path to save model
        path_to_dict: (optional) path to save dictionary instance
        """

        path_to_model = os.path.join(self.__root_dir, data_name, 'lda_model')
        path_to_dict = os.path.join(self.__root_dir, data_name, 'lda_dict')

        if not os.path.exists(path_to_model):
            raise ValueError('no model found in %s' % path_to_model)

        if not os.path.exists(path_to_dict):
            raise ValueError('no model found in %s' % path_to_dict)

        if self.__logger:
            self.__logger.info('load model')
        self.__model = gensim.models.ldamodel.LdaModel.load(path_to_model)
        self.__dict = gensim.corpora.Dictionary.load_from_text(path_to_dict)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])

    def save_model(self, data_name: str):
        os.makedirs(os.path.join(self.__root_dir, data_name), exist_ok=True)
        self.__model.save(os.path.join(self.__root_dir, data_name, 'lda_model'))
        self.__dict.save_as_text(os.path.join(self.__root_dir, data_name, 'lda_dict'))

    def train(self,
              data: list,
              cleaning: bool = True,
              num_topics: int = 15,
              data_name: str = None):
        """ train topic model

         Parameter
        ----------------
        data: list of document (list of string)
        num_topics: number of topic
        cleaning: If False, don't apply cleaning filter to string
        path_to_model: (optional) path to save model
        path_to_dict: (optional) path to save dictionary instance

        """
        if cleaning:
            if self.__logger:
                self.__logger.info("cleaning data: length %i" % len(data))
            processor = Processing(language=self.__lan)
            data = [processor(d, return_token=True)[1] for d in data]
        else:
            data = [d.split() for d in data]

        # build LDA model
        if self.__logger:
            self.__logger.info("building corpus")
        self.__dict = corpora.Dictionary(data)
        corpus = [self.__dict.doc2bow(text) for text in data]
        if self.__logger:
            self.__logger.info("training model")
        self.__model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                       num_topics=num_topics,
                                                       id2word=self.__dict)
        if self.__logger:
            self.__logger.info("save model and corpus")
        if data_name is not None:
            self.save_model(data_name)

    def distribution_word_topic(self,
                                topic_id: int,
                                num: int = 5,
                                return_word_id: bool = False):
        """ word probability distribution of given topic: p(w|z_i) with i = topic_id

         Parameter
        ----------------
        topic_id: topic id
        num: number of word to get
        return_word_id: return word_id instead of word if True

         Return
        --------------
        [(word_0, prob_0), ..., (word_num, prob_num)] in order of probability
        """
        if not self.if_ready:
            raise ValueError('model is not ready')

        if topic_id > self.topic_size:
            raise ValueError('topic_id should be less than topic number %i' % self.topic_size)

        if return_word_id:
            return self.__model.get_topic_terms(topic_id, num)
        else:
            return self.__model.show_topic(topic_id, num)

    def distribution_word_document(self,
                                   tokens: int,
                                   return_word_id: bool = False):
        """ word probability distribution of given document: p(w|d) = p(w|z)p(z|d) with d=tokens

         Parameter
        ----------------
        tokens: document to sample word
        return_word_id: return word_id instead of word if True

         Return
        --------------
        [(word_0, prob_0), ..., (word_n, prob_n)] in order of probability
        Note that `n` is dynamically changing based on the coverage of probability and the probability itself will
        change slightly due to the randomness of sampling
        """
        if not self.if_ready:
            raise ValueError('model is not ready')

        bow = self.__dict.doc2bow(tokens)
        if return_word_id:
            return self.__model[bow]
        else:
            return [(self.__dict.id2token[_id], prob) for _id, prob in self.__model[bow]]

    def distribution_topic_document(self, tokens: list):
        """ topic probability distribution of given documents p(z|d = tokens)

         Parameter
        ----------------
        tokens: document to get topic distribution

         Return
        -----------
        [(topic_id, prob), (topic_id, prob), ...] in order of prob
        """
        if not self.if_ready:
            raise ValueError('model is not ready')

        bow = self.__dict.doc2bow(tokens)
        topic_dist = self.__model.get_document_topics(bow, minimum_probability=0.0)
        return topic_dist

        # bow = self.__dict.doc2bow(list(self.__dict.id2token.values()))
        # self.__model.get_document_topics(bow, minimum_probability=0.0)

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
        if not self.if_ready:
            raise ValueError('model is not ready')

        topic_table = self.__model.get_topics()
        if tokens is None:
            return topic_table
        else:
            ids = [self.__dict.token2id[t] for t in tokens]
            topic_dist = topic_table[:, ids]
            return topic_dist

    @property
    def model(self):
        """ gensim model instance of LDA """
        if not self.if_ready:
            raise ValueError('model is not ready')
        return self.__model

    @property
    def vocab_size(self):
        """ vocabulary size """
        if not self.if_ready:
            raise ValueError('model is not ready')
        return len(self.__dict.token2id)

    @property
    def vocab(self):
        """ vocabulary size """
        if not self.if_ready:
            raise ValueError('model is not ready')
        return list(self.__dict.token2id.keys())

    @property
    def topic_size(self):
        """ topic size """
        if not self.if_ready:
            raise ValueError('model is not ready')
        return self.__model.num_topics

    @property
    def dictionary(self):
        """ gensim dictionary instance """
        if not self.if_ready:
            raise ValueError('model is not ready')
        return self.__dict

    @property
    def if_ready(self):
        if self.__dict is None or self.__model is None:
            return False
        else:
            return True
