""" Model to utilize TFIDF: Wrapper of gensim API for TFIDF

WARNING: only for English!!!

"""

import os
import gensim
from gensim import corpora
from .processing import Processing
from ..util import create_log

DEFAULT_ROOD_DIR = os.path.join(os.path.expanduser("~"), 'keyphraser_data')


class TFIDF:
    """ calculate TFIDF matrix for given dataset, and instance to utilize it

     Usage
    ------------------
    >>> docs = [
    'doc1\nmore\nmore',
    'doc2\nmore\nmore',
    'doc3\nmore\nmore']
    >>> tfidf = TFIDF()
    # train model
    >>> tfidf.train(docs, path_to_model='test_model', path_to_dict='test_dict')
    # or load model
    >>> tfidf.load_lda(path_to_model='test_model', path_to_dict='test_dict')
    """

    def __init__(self,
                 root_dir: str = None,
                 debug: bool = False,
                 language: str = 'en'):
        self.__lan = language
        self.__dict = None
        self.__model = None
        self.__logger = create_log() if debug else None
        if root_dir is None:
            self.__root_dir = os.path.join(DEFAULT_ROOD_DIR, 'tfidf')
        else:
            self.__root_dir = os.path.join(root_dir, 'tfidf')

    def load(self, data_name: str):
        """ load saved lda model and dictionary instance used to train the model

         Parameter
        ----------------
        path_to_model: (optional) path to save model
        path_to_dict: (optional) path to save dictionary instance
        """

        path_to_model = os.path.join(self.__root_dir, data_name, 'tfidf_model')
        path_to_dict = os.path.join(self.__root_dir, data_name, 'tfidf_dict')

        if not os.path.exists(path_to_model):
            raise ValueError('no model found in %s' % path_to_model)

        if not os.path.exists(path_to_dict):
            raise ValueError('no model found in %s' % path_to_dict)

        if self.__logger:
            self.__logger.info('load model')
        self.__model = gensim.models.TfidfModel.load(path_to_model)
        self.__dict = gensim.corpora.Dictionary.load_from_text(path_to_dict)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])

    def save_model(self, data_name: str):
        os.makedirs(os.path.join(self.__root_dir, data_name), exist_ok=True)
        self.__model.save(os.path.join(self.__root_dir, data_name, 'tfidf_model'))
        self.__dict.save_as_text(os.path.join(self.__root_dir, data_name, 'tfidf_dict'))

    def train(self,
              data: list,
              cleaning: bool = True,
              normalize: bool = True,
              data_name: str = None):
        """ train topic model

         Parameter
        ----------------
        data: list of document (list of string)
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
            data = [d if type(d) is list else d.split() for d in data]

        # build model
        if self.__logger:
            self.__logger.info("building corpus")
        self.__dict = corpora.Dictionary(data)
        self.__dict.id2token = dict([(v, k) for k, v in self.__dict.token2id.items()])
        corpus = [self.__dict.doc2bow(text) for text in data]
        if self.__logger:
            self.__logger.info("training model")
        self.__model = gensim.models.TfidfModel(corpus=corpus, normalize=normalize)
        if self.__logger:
            self.__logger.info("save model and corpus")
        if data_name is not None:
            self.save_model(data_name)

    def distribution_word(self,
                          tokens: list,
                          return_word_id: bool = False):
        """ word distribution of given document with pre-calculate TFIDF matrix

         Parameter
        ----------------
        tokens: document to get topic distribution

         Return
        -----------
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
