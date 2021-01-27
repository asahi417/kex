""" Implementation of TopicRank """
from itertools import chain, combinations, product

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from ._phrase_constructor import PhraseConstructor

__all__ = 'TopicRank'


class TopicRank:
    """ TopicRank """

    def __init__(self,
                 language: str = 'en',
                 random_prob: float = 0.85,
                 tol: float = 0.0001,
                 clustering_threshold: float = 0.74,
                 linkage_method: str = 'average'):
        """ TopicRank

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
        clustering_threshold: float
            threshold for clustering algorithm
        linkage_method: str
            graph linkage method
        add_verb: bool
            take verbs into account
        no_stemming: bool
            no stemming is applied (True if the document is already stemmed)
        """
        self.prior_required = False
        self.__random_prob = random_prob
        self.__tol = tol

        self.__linkage_method = linkage_method
        self.__clustering_threshold = clustering_threshold

        self.phrase_constructor = PhraseConstructor(language=language)

    def topic_clustering(self, stemmed_phrases: list):
        """ grouping given phrases to topic based on HAC by there tokens

         Parameter
        --------------------
        stemmed_phrases: list
            list of stemmed keywords

         Return
        --------------------
        grouped_phrase: list
            grouped keywords
        """
        unique_token = set(chain(*[p.split() for p in stemmed_phrases]))
        token_to_id = dict([(t, i) for i, t in enumerate(unique_token)])

        # convert phrases to vector spanned by unique token
        matrix = np.zeros((len(stemmed_phrases), len(unique_token)))
        for n, p in enumerate(stemmed_phrases):
            indices = [token_to_id[_p] for _p in set(p.split())]
            matrix[n, indices] = 1

        # calculate distance
        distance = pdist(matrix, 'jaccard')
        # compute the clusters
        links = linkage(distance, method=self.__linkage_method)
        # get cluster id: len(clusters) == len(stemmed_phrases)
        clusters = fcluster(links, t=self.__clustering_threshold, criterion='distance')
        grouped_phrase = [np.array(stemmed_phrases)[clusters == c_id].tolist() for c_id in set(clusters)]
        return grouped_phrase

    def build_graph(self, document: str):
        """ Build basic graph with Topic """

        # convert phrase instance
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keywords
            return None

        # group phrases as topic classes
        grouped_phrases = self.topic_clustering(list(phrase_instance.keys()))
        group_id = list(range(len(grouped_phrases)))

        # initialize graph instance
        graph = nx.Graph()

        # add nodes
        graph.add_nodes_from(group_id)

        def offset_distance(a: list, b: list):
            return 1/(a[0] - b[1] + 1) if a[0] > b[0] else 1/(b[0] - a[1] + 1)

        # add edges with weight
        for id_x, id_y in combinations(group_id, 2):
            # list of offset
            x_offsets = list(chain(*[phrase_instance[_x_phrase]['offset'] for _x_phrase in grouped_phrases[id_x]]))
            y_offsets = list(chain(*[phrase_instance[_y_phrase]['offset'] for _y_phrase in grouped_phrases[id_y]]))

            weight = sum([offset_distance(_x, _y) for _x, _y in product(x_offsets, y_offsets)])
            graph.add_edge(id_x, id_y, weight=weight)
        return graph, phrase_instance, grouped_phrases, len(stemmed_tokens)

    def run_pagerank(self, graph):
        node_score = nx.pagerank(G=graph, alpha=self.__random_prob, tol=self.__tol, weight='weight')
        return node_score

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
        # make graph and get data structure for candidate phrase
        output = self.build_graph(document)
        if output is None:
            return []

        graph, phrase_instance, grouped_phrases, original_sentence_token_size = output
        # pagerank to get score for individual word (sum of score will be one)
        node_score = self.run_pagerank(graph)

        # combine score to get score of phrase
        def aggregation(group_id, phrases):
            first_phrase_id = np.argmin([phrase_instance[p]['offset'][0][0] for p in phrases])
            phrase = phrases[first_phrase_id]
            return phrase, node_score[group_id]

        phrase_score = [aggregation(group_id, phrases) for group_id, phrases in enumerate(grouped_phrases)]

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
