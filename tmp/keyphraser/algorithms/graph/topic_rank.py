"""TopicRank keyphrase extraction model.
Implementation of the TopicRank model for keyword extraction.
"""

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from itertools import chain, combinations, product
from ..processing import Processing


class TopicRank:
    """TopicRank

     Usage
    -----------------
    >>> model = TextRank(window_size=10)
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
                 language: str='en',
                 random_prob: float=0.85,
                 tol: float=0.0001,
                 clustering_threshold: float=0.74,
                 linkage_method: str='average',
                 add_verb: bool=False
                 ):
        """ TopicRank algorithm

        :param stop_words: Additional stopwords (As default, NLTK English stopwords is contained)
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param clustering_threshold: parameter for clustering (default should is the one used in original paper)
        :param linkage_method: parameter for clustering (default should is the one used in original paper)
        :param tol: PageRank parameter that define tolerance of convergence
        """
        self.__random_prob = random_prob
        self.__tol = tol

        self.__linkage_method = linkage_method
        self.__clustering_threshold = clustering_threshold

        self.processor = Processing(language=language, add_verb=add_verb)
        self.list_of_graph = []
        self.list_of_phrase = []
        self.list_of_group = []
        self.directed_graph = False
        self.weighted_graph = True

    def topic_clustering(self, stemmed_phrases: list):
        """ grouping given phrases to topic based on HAC by there tokens

        :param stemmed_phrases: list of stemmed phrases
        :return: list of list corresponding to each topic
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

    def extract(self,
                target_documents: list,
                count: int = 10):
        """ Extract keyphrases from list of documents

        :param target_documents: list of target document
        :param count: number of phrase to get
        :return: list of candidate phrase corresponding to each target document
            [
                [('aa', 0.5, language_attribution), ('b', 0.3, language_attribution), ...], ...
            ]
        """
        self.list_of_graph = []
        self.list_of_phrase = []
        self.list_of_group = []
        # convert to string
        values = [self.get_phrase_single_document(doc, count=count)
                  if len(str(doc)) > 2 and type(doc) is str else [] for doc in target_documents]
        return values

    def build_graph(self, target_document: str):
        """ build graph with topics """

        # convert phrase instance
        phrase_instance, cleaned_document_tokenized = self.processor(target_document, return_token=True)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
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

        original_sentence_token_size = len(cleaned_document_tokenized)
        return graph, phrase_instance, grouped_phrases, original_sentence_token_size

    def run_pagerank(self, graph):
        node_score = nx.pagerank(G=graph,
                                 alpha=self.__random_prob,
                                 tol=self.__tol,
                                 weight='weight')
        return node_score

    def get_phrase_single_document(self,
                                   target_document: str,
                                   count: int = 10):
        """ Extract keyphrase from single document

        :param target_document: target document
        :param count: number of phrase to get
        :return: list of candidate phrase with score such as [('aa', 0.5), ('b', 0.3), ..]
        """

        # make graph and get data structure for candidate phrase
        output = self.build_graph(target_document)
        if output is None:
            return []

        graph, phrase_instance, grouped_phrases, original_sentence_token_size = output
        self.list_of_graph.append(graph)
        self.list_of_group.append(grouped_phrases)
        self.list_of_phrase.append(phrase_instance)

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
        count_valid = min(count, len(phrase_score_sorted_list))

        def modify_output(stem, score):
            tmp = phrase_instance[stem]
            tmp['score'] = score
            tmp['original_sentence_token_size'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val
