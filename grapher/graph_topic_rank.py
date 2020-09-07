""" Implementation of TopicRank """
from itertools import chain, combinations, product

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from ._phrase_constructor import PhraseConstructor

__all__ = 'TopicRank'


class TopicRank:
    """TopicRank

     Usage
    -----------------
    >>> model = TopicRank()
    >>> sample =
    'We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier '
    'detection. It starts by training word embeddings on the target document to capture semantic regularities among '
    'the words. It then uses the minimum covariance determinant estimator to model the distribution of non-keyphrase '
    'word vectors, under the assumption that these vectors come from the same distribution, indicative of their '
    'irrelevance to the semantics expressed by the dimensions of the learned vector representation. Candidate '
    'keywords only consist of words that are detected as outliers of this dominant distribution. Empirical results '
    'show that our approach outperforms stateof-the-art and recent unsupervised keyphrase extraction methods.'
    >>> model.get_keywords(sample)
    """

    def __init__(self,
                 language: str = 'en',
                 random_prob: float = 0.85,
                 tol: float = 0.0001,
                 clustering_threshold: float = 0.74,
                 linkage_method: str = 'average',
                 add_verb: bool = False):
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
        """
        self.__random_prob = random_prob
        self.__tol = tol

        self.__linkage_method = linkage_method
        self.__clustering_threshold = clustering_threshold

        self.phrase_constructor = PhraseConstructor(language=language, add_verb=add_verb)

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
        phrase_instance, tokens = self.phrase_constructor.get_phrase(document)
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
            return sum([1/abs(i[0] - i[1]) for i in product(range(*a), range(*b))])

        # add edges with weight
        for id_x, id_y in combinations(group_id, 2):
            # list of offset
            x_offsets = list(chain(*[phrase_instance[_x_phrase]['offset'] for _x_phrase in grouped_phrases[id_x]]))
            y_offsets = list(chain(*[phrase_instance[_y_phrase]['offset'] for _y_phrase in grouped_phrases[id_y]]))
            weight = sum([offset_distance(_x, _y) for _x, _y in product(x_offsets, y_offsets)])
            graph.add_edge(id_x, id_y, weight=weight)

        return graph, phrase_instance, grouped_phrases, len(tokens)

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
        a list of keyphrase with score eg) [('aa', 0.5), ('b', 0.3), ..]
        """
        # make graph and get data structure for candidate phrase
        output = self.build_graph(document)
        if output is None:
            return []

        graph, phrase_instance, grouped_phrases, original_sentence_token_size = output

        # pagerank to get score for individual word (sum of score will be one)
        node_score = self.run_pagerank(graph)

        # combine score to get score of phrase
        phrase_score_dict = dict()
        for group_id, phrases in enumerate(grouped_phrases):
            first_phrase_id = np.argmin([phrase_instance[p]['offset'][0][0] for p in phrases])
            phrase = phrases[first_phrase_id]
            phrase_score_dict[phrase] = node_score[group_id]

        # sorting
        phrase_score_sorted_list = sorted(phrase_score_dict.items(), key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['raw'] = tmp['raw'][0]
            tmp['lemma'] = tmp['lemma'][0]
            tmp['n_source_tokens'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val