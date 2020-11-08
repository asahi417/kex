""" Implementation of basic TextRank variants
 * TextRank: Undirected Unweighted Graph
 * SingleRank: Undirected Weighted Graph (weight is co-occurrence)
 * PositionRank: Undirected Weighted Graph (weight is the offset)
 """
import networkx as nx
from itertools import chain

import numpy as np
from ._phrase_constructor import PhraseConstructor

__all__ = ('TextRank', 'SingleRank', 'PositionRank')


class TextRank:
    """ TextRank

     Usage
    -----------------
    >>> model = TextRank()
    >>> sample =
    'We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier '
    'detection. It starts by training word embeddings on the target document to capture semantic regularities among '
    'the words. It then uses the minimum covariance determinant estimator to model the distribution of non-keyphrase '
    'word vectors, under the assumption that these vectors come from the same distribution, indicative of their '
    'irrelevance to the semantics expressed by the dimensions of the learned vector representation. Candidate '
    'keyphrases only consist of words that are detected as outliers of this dominant distribution. Empirical results '
    'show that our approach outperforms stateof-the-art and recent unsupervised keyphrase extraction methods.'
    >>> model.get_keywords(sample)
    """

    def __init__(self, language: str = 'en', window_size: int = 10, random_prob: float = 0.85, tol: float = 0.0001):
        """ TextRank

         Parameter
        ------------------------
        language: str
            `en` or `ja`
        window_size: int
            window size to make graph edges
        random_prob: float
            PageRank parameter that is coefficient of convex combination of random suffer model
        tol: float
            PageRank parameter that define tolerance of convergence
        """
        self.__window_size = window_size
        self.__random_prob = random_prob
        self.__tol = tol

        self.phrase_constructor = PhraseConstructor(language=language)
        self.weighted_graph = False
        self.direct_graph = False
        self.prior_required = False

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
        # make graph and get data structure for candidate phrase
        output = self.build_graph(document)
        if output is None:
            return []

        graph, phrase_instance, original_sentence_token_size = output
        node_score = self.run_pagerank(graph)

        # combine score to get score of phrase
        phrase_score = [
            (candidate_phrase_stemmed_form, sum(node_score[t] for t in candidate_phrase_stemmed_form.split()))
            for candidate_phrase_stemmed_form in phrase_instance.keys()
        ]

        # sorting
        phrase_score_sorted_list = sorted(phrase_score, key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['n_source_tokens'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val

    def build_graph(self, document: str):
        """ Build basic graph
        - nodes: phrases extracted from given document
        - edge: co-occurrence in certain window
        - weight: count of co-occurrence

         Parameter
        ------------------------
        document: str
            single document

         Return
        ------------------------
        graph:
            graph instance
        phrase_instance:
            phrase instance
        original_sentence_token_size: int
            token size in target_document
        """

        # convert phrase instance
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
            return None

        # initialize graph instance
        if self.direct_graph:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # add nodes
        unique_tokens_in_candidate = list(set(chain(*[i.split() for i in phrase_instance.keys()])))
        graph.add_nodes_from(unique_tokens_in_candidate)

        # add edges
        for position, __start_node in enumerate(stemmed_tokens[:-self.__window_size]):

            # ignore invalid token
            if __start_node not in unique_tokens_in_candidate:
                continue

            for __position in range(position, position + self.__window_size):
                __end_node = stemmed_tokens[__position]

                # ignore invalid token
                if __end_node not in unique_tokens_in_candidate:
                    continue

                if not graph.has_edge(__start_node, __end_node):
                    graph.add_edge(__start_node, __end_node, weight=1.0)
                else:
                    if self.weighted_graph:
                        # SingleRank employ weight as the co-occurrence times
                        graph[__start_node][__end_node]['weight'] += 1.0

        return graph, phrase_instance, len(stemmed_tokens)

    def run_pagerank(self, graph, personalization=None):
        if personalization:
            return nx.pagerank(
                G=graph, alpha=self.__random_prob, tol=self.__tol, weight='weight', personalization=personalization)
        else:
            return nx.pagerank(G=graph, alpha=self.__random_prob, tol=self.__tol, weight='weight')


class SingleRank(TextRank):
    """ SingleRank """

    def __init__(self, *args, **kwargs):
        """ SingleRank """

        super(SingleRank, self).__init__(*args, **kwargs)
        self.weighted_graph = True


class PositionRank(TextRank):
    """ PositionRank """

    def __init__(self, *args, **kwargs):
        """ PositionRank """
        super(PositionRank, self).__init__(*args, **kwargs)
        self.weighted_graph = True

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
        # make graph and get data structure for candidate phrase
        output = self.build_graph(document)
        if output is None:
            return []

        graph, phrase_instance, original_sentence_token_size = output

        # calculate bias
        normalizer = np.sum([np.sum([1/(1+s) for s, _ in v['offset']]) for v in phrase_instance.values()])
        bias = dict([
            (k, np.sum([1 / (1 + s) for s, _ in v['offset']]) / normalizer) for k, v in phrase_instance.items()
        ])

        # pagerank to get score for individual word (sum of score will be one)
        node_score = self.run_pagerank(graph, bias)

        # combine score to get score of phrase
        phrase_score = [
            (candidate_phrase_stemmed_form, sum(node_score[t] for t in candidate_phrase_stemmed_form.split()))
            for candidate_phrase_stemmed_form in phrase_instance.keys()
        ]

        # sorting
        phrase_score_sorted_list = sorted(phrase_score, key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['n_source_tokens'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val

