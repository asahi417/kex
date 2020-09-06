"""ExpandRank keyphrase extraction model.
Implementation of the ExpandRank model for keyword extraction.
"""

import os
from itertools import combinations
from .text_rank import TextRank
from ..tfidf import TFIDF


class ExpandRank(TextRank):
    """ExpandRank for keyword extraction.

    As neighbouring information, we use TFIDF-based similarity used in the original paper.

     Usage
    -----------------
    >>> model = ExpandRank(window_size=10)
    >>> docs = [
    'doc1\nmore\nmore',
    'doc2\nmore\nmore',
    'doc3\nmore\nmore'
    ]
    >>> model.extract(docs, count=1)
[[('novel unsupervis keyphras extract approach',
   0.2573069513776437,
   {'stemmed': 'novel unsupervis keyphras extract approach',
    'pos': 'ADJ ADJ NOUN NOUN NOUN',
    'raw': ['novel unsupervised keyphrase extraction approach'],
    'lemma': ['novel unsupervised keyphrase extraction approach'],
    'offset': [[3, 7]],
    'count': 1})],
 [('art graph',
   0.24564816718881646,
   {'stemmed': 'art graph',
    'pos': 'NOUN NOUN',
    'raw': ['art graph'],
    'lemma': ['art graph'],
    'offset': [[71, 72]],
    'count': 1})],
 [('local word embed',
   0.11093480422106389,
   {'stemmed': 'local word embed',
    'pos': 'ADJ NOUN NOUN',
    'raw': ['local word embeddings'],
    'lemma': ['local word embedding'],
    'offset': [[43, 45]],
    'count': 1})]]
    >>> model.list_of_graph  # check graphs
    >>> model.similarity  # check similarity
    """

    def __init__(self,
                 window_size: int = 10,
                 stop_words: list = None,
                 random_prob: float = 0.85,
                 tol: float = 0.0001,
                 num_neighbour: int = 5):
        """ ExpandRank algorithm

        :param window_size: Length of window to make edges
        :param stop_words: Additional stopwords (As default, NLTK English stopwords is contained)
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param tol: PageRank parameter that define tolerance of convergence
        """

        super(ExpandRank, self).__init__(window_size=window_size,
                                         stop_words=stop_words,
                                         random_prob=random_prob,
                                         tol=tol)
        self.similarity = []
        self.tfidf = TFIDF()
        self.weighted_graph = True
        # TODO: original paper constrains the number of neighbourhood
        self.__num_neighbour = num_neighbour

    def setup_tfidf(self,
                    path_to_model: str,
                    path_to_dict: str,
                    data: list = None,
                    normalize: bool = True):
        """ setup lda model (should be run before keyphrase extraction) """

        if os.path.exists(path_to_model) and os.path.exists(path_to_dict):
            self.tfidf.load(path_to_model=path_to_model, path_to_dict=path_to_dict)
        else:
            if data is None:
                raise ValueError('No pretrained LDA checkpoints found. provide data to train LDA')
            self.tfidf.train(data=data,
                             normalize=normalize,
                             path_to_model=path_to_model,
                             path_to_dict=path_to_dict)

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
        self.similarity = []
        self.list_of_graph = []
        self.list_of_phrase = []

        # get neighbouring information (similarity among docs based on TFIDF vector's cosine dist)
        def get_cleaned_sentence(target_str):
            _, cleaned_document_tokenized = self.processor(target_str, return_token=True)
            return ' '.join(cleaned_document_tokenized)
        sents = [get_cleaned_sentence(single_doc) for single_doc in target_documents]
        similarity = self.tfidf_similarity(sents)
        self.similarity = similarity

        # construct initial graph
        list_of_graph_and_phrase = [list(self.build_graph(doc, weighted_graph=True)) for doc in target_documents]
        list_of_graph_and_phrase_new = list_of_graph_and_phrase.copy()

        # re-weight graph
        for comb, sim in similarity.items():
            graph_x = list_of_graph_and_phrase[comb[0]][0]
            graph_y = list_of_graph_and_phrase[comb[1]][0]

            edge_x = set(graph_x.edges)
            edge_y = set(graph_y.edges)
            edge_intersection = edge_x.intersection(edge_y)
            if len(edge_intersection) == 0:
                continue

            graph_x_new = list_of_graph_and_phrase_new[comb[0]][0]
            graph_y_new = list_of_graph_and_phrase_new[comb[1]][0]

            for node_in, node_out in edge_intersection:
                # note that this graph is undirected so `in`, `out` actually don't have any meanings
                graph_x_new[node_in][node_out]['weight'] += sim * graph_y[node_in][node_out]['weight']
                graph_y_new[node_in][node_out]['weight'] += sim * graph_x[node_in][node_out]['weight']

            list_of_graph_and_phrase_new[comb[0]][0] = graph_x_new
            list_of_graph_and_phrase_new[comb[1]][0] = graph_y_new

        self.list_of_graph = [graph for graph, phrase in list_of_graph_and_phrase_new]
        self.list_of_phrase = [phrase for graph, phrase in list_of_graph_and_phrase_new]

        def page_rank(graph, phrase_instance):
            # pagerank to get score for individual word (sum of score will be one)
            node_score = self.run_pagerank(graph)
            # combine score to get score of phrase
            phrase_score_dict = dict()
            for candidate_phrase_stemmed_form in phrase_instance.keys():
                tokens_in_phrase = candidate_phrase_stemmed_form.split()
                phrase_score_dict[candidate_phrase_stemmed_form] = sum([node_score[t] for t in tokens_in_phrase])

            # sorting
            phrase_score_sorted_list = sorted(phrase_score_dict.items(),
                                              key=lambda key_value: key_value[1],
                                              reverse=True)
            count_valid = min(count, len(phrase_score_sorted_list))
            val = [(stem, score, phrase_instance[stem]) for stem, score in phrase_score_sorted_list[:count_valid]]
            return val

        return [page_rank(graph, phrase) for graph, phrase in list_of_graph_and_phrase_new]

    def tfidf_similarity(self, list_of_docs: list):
        """calculate tfidf for each document and compute cosine similarity of each tfidf vector"""

        if len(set(list_of_docs)) == 1:
            return dict([((0, 0), 1.0)])

        def product_tfidf_vector(x: list, y: list):
            x = dict(x)
            y = dict(y)
            inter_section = set(x.keys()).intersection(set(y.keys()))
            if len(inter_section) == 0:
                return 0.0
            inner_prod = sum([x[i] * y[i] for i in inter_section])
            squared_norm_x = sum([_x * _x for _x in x.values()])
            squared_norm_y = sum([_y * _y for _y in y.values()])
            cos_sim = inner_prod/((squared_norm_x * squared_norm_y) ** 0.5 + 1e-7)
            return cos_sim

        similarity = dict([
            (
                comb, product_tfidf_vector(
                    self.tfidf.distribution_word(list_of_docs[comb[0]].split()),
                    self.tfidf.distribution_word(list_of_docs[comb[1]].split())
                )
            )
            for comb in combinations(list(range(len(list_of_docs))), 2)
        ])

        return similarity
