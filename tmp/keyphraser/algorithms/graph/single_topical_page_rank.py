"""TopicalPageRank keyphrase extraction model.
Implementation of the TopicalPageRank model for keyword extraction.

* Single TopicalPageRank
"""

import numpy as np
from .topical_page_rank import TopicalPageRank


class SingleTPR(TopicalPageRank):
    """SingleTopicalPageRank

     Usage
    -----------------
    >>> model = TopicalPageRank(window_size=10)
    >>> sample =
    'We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier '
    'detection. It starts by training word embeddings on the target document to capture semantic regularities among '
    'the words. It then uses the minimum covariance determinant estimator to model the distribution of non-keyphrase '
    'word vectors, under the assumption that these vectors come from the same distribution, indicative of their '
    'irrelevance to the semantics expressed by the dimensions of the learned vector representation. Candidate '
    'keyphrases only consist of words that are detected as outliers of this dominant distribution. Empirical results '
    'show that our approach outperforms stateof-the-art and recent unsupervised keyphrase extraction methods.'
    >>> model.extract([sample], count=2)
    [[
        ('novel unsupervis keyphras extract approach', 0.2570445598381217,
           {'stemmed': 'novel unsupervis keyphras extract approach',
            'pos': 'ADJ ADJ NOUN NOUN NOUN',
            'raw': ['novel unsupervised keyphrase extraction approach'],
            'lemma': ['novel unsupervised keyphrase extraction approach'],
            'offset': [[3, 7]],
            'count': 1}),
        ('keyphras word vector', 0.19388916977478182,
           {'stemmed': 'keyphras word vector',
            'pos': 'NOUN NOUN NOUN',
            'raw': ['keyphrase word vectors'],
            'lemma': ['keyphrase word vector'],
            'offset': [[49, 51]],
            'count': 1})
    ]]
    >>> model.list_of_graph  # check graphs
    """

    def __init__(self,
                 path_to_dict: str,
                 path_to_model: str,
                 window_size: int = 10,
                 stop_words: list = None,
                 random_prob: float = 0.85,
                 tol: float = 0.0001
                 ):
        """

        :param window_size: Length of window to make edges
        :param stop_words: Additional stopwords (As default, NLTK English stopwords is contained)
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param tol: PageRank parameter that define tolerance of convergence
        """

        super(SingleTPR, self).__init__(path_to_dict=path_to_dict,
                                        path_to_model=path_to_model,
                                        window_size=window_size,
                                        stop_words=stop_words,
                                        random_prob=random_prob,
                                        tol=tol)
        self.weighted_graph = True

    def get_phrase_single_document(self,
                                   target_document: str,
                                   count: int = 10):
        """ Extract keyphrase from single document

        :param target_document: target document
        :param count: number of phrase to get
        :return: list of candidate phrase with score such as [('aa', 0.5), ('b', 0.3), ..]
        """

        if not self.lda.if_ready:
            raise ValueError('no TopicModel setup')

        # make graph and get data structure for candidate phrase
        graph, phrase_instance, cleaned_document_tokenized \
            = self.build_graph(target_document, directed_graph=True, return_cleaned_document=True)
        self.list_of_graph.append(graph)
        self.list_of_phrase.append(phrase_instance)

        # pagerank to get score for individual word (sum of score will be one)
        word_matrix = self.lda.probability_word()  # topic size x vocab
        vocab = self.lda.dictionary.token2id
        topic_dist = self.lda.distribution_topic_document(cleaned_document_tokenized)
        topic_vector = np.array([i for t, i in topic_dist])  # topic size

        # single TPR

        def norm_inner(a, b):
            return np.sum(a * b)/(np.sqrt(np.sum(a*a)+np.sum(b*b))+1e-7)

        bias = dict()
        for word in graph.nodes():
            if word in vocab.keys():
                word_id = vocab[word]
                word_vector = word_matrix[:, word_id]
                bias[word] = norm_inner(word_vector, topic_vector)
            else:
                bias[word] = 0.0

        # Normalize the topical word importance of words
        norm = sum(bias.values())
        for word in bias:
            bias[word] /= norm

        self.list_of_bias.append(bias)
        node_score = self.run_pagerank(graph, personalization=bias)

        # combine score to get score of phrase
        phrase_score_dict = dict()
        for candidate_phrase_stemmed_form in phrase_instance.keys():
            tokens_in_phrase = candidate_phrase_stemmed_form.split()
            phrase_score_dict[candidate_phrase_stemmed_form] = sum([node_score[t] for t in tokens_in_phrase])

        # sorting
        phrase_score_sorted_list = sorted(phrase_score_dict.items(), key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(count, len(phrase_score_sorted_list))
        val = [(stem, score, phrase_instance[stem]) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val
