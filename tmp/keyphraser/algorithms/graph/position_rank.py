"""PositionRank keyphrase extraction model.

Implementation of the PositionRank (and its variants) model for keyword extraction described in:
"""

import numpy as np
from .text_rank import TextRank


class PositionRank(TextRank):
    """TextRank and its variants for keyword extraction.

     Usage
    -----------------
    >>> model = PositionRank(window_size=10)
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
                 language: str = 'en',
                 window_size: int = 10,
                 random_prob: float = 0.85,
                 tol: float = 0.0001):
        """ TextRank algorithm

        :param window_size: Length of window to make edges
        :param stop_words: Additional stopwords (As default, NLTK English stopwords is contained)
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param tol: PageRank parameter that define tolerance of convergence
        """

        super(PositionRank, self).__init__(language=language,
                                           window_size=window_size,
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

        # make graph and get data structure for candidate phrase
        output = self.build_graph(target_document, weighted_graph=True)
        if output is None:
            return []

        graph, phrase_instance, original_sentence_token_size = output
        self.list_of_graph.append(graph)
        self.list_of_phrase.append(phrase_instance)

        # calculate bias
        normalizer = np.sum([np.sum([1/(1+s) for s, _ in v['offset']]) for v in phrase_instance.values()])
        bias = dict([
            (k, np.sum([1 / (1+s) for s, _ in v['offset']])/normalizer) for k, v in phrase_instance.items()
        ])

        # pagerank to get score for individual word (sum of score will be one)
        node_score = self.run_pagerank(graph, personalization=bias)

        # combine score to get score of phrase
        phrase_score_dict = dict()
        for candidate_phrase_stemmed_form in phrase_instance.keys():
            tokens_in_phrase = candidate_phrase_stemmed_form.split()
            phrase_score_dict[candidate_phrase_stemmed_form] = sum([node_score[t] for t in tokens_in_phrase])

        # sorting
        phrase_score_sorted_list = sorted(phrase_score_dict.items(), key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(count, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['original_sentence_token_size'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]

        return val
