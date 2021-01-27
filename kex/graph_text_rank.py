""" Implementation of basic TextRank variants
 * TextRank: Undirected Unweighted Graph
 * SingleRank: Undirected Weighted Graph (weight is co-occurrence)
 * PositionRank: Undirected Weighted Graph (weight is co-occurrence, bias is offset)
 * TFIDFRank: Undirected Weighted Graph (weight is co-occurrence, bias is TFIDF word probability)
 * LexRank: Undirected Weighted Graph (weight is co-occurrence, bias is lexical specificity)
 * SingleTPR: Directed Weighted Graph (weight is co-occurrence, bias is LDA topic x word distribution)
"""
import numpy as np
import networkx as nx
from itertools import chain

from ._phrase_constructor import PhraseConstructor
from .tfidf import TFIDF
from .lda import LDA
from .lexical_specificity import LexSpec

__all__ = ('TextRank', 'SingleRank', 'PositionRank', 'TFIDFRank', 'SingleTPR', 'LexRank')


class TextRank:
    """ TextRank """

    def __init__(self,
                 language: str = 'en',
                 window_size: int = 2,
                 random_prob: float = 0.85,
                 tol: float = 0.0001):
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
        self.normalize = True
        self.page_rank_bias_type = None
        self.tfidf = None
        self.lexical = None
        self.lda = None

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

        graph, phrase_instance, original_sentence_token_size, bias = output
        node_score = self.run_pagerank(graph, personalization=bias)

        # combine score to get score of phrase
        phrase_score = [
            (
                candidate_phrase_stemmed_form,
                sum(node_score[t] for t in candidate_phrase_stemmed_form.split())
                / len(candidate_phrase_stemmed_form.split())  # - v['offset'][0][0] * 1e-8
                if self.normalize else sum(node_score[t] for t in candidate_phrase_stemmed_form.split())
            ) for candidate_phrase_stemmed_form, v in phrase_instance.items()
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
        for position, __start_node in enumerate(stemmed_tokens):

            # ignore invalid token
            if __start_node not in unique_tokens_in_candidate:
                continue

            for __position in range(position, min(position + self.__window_size, len(stemmed_tokens))):
                __end_node = stemmed_tokens[__position]

                # ignore invalid token
                if __end_node not in unique_tokens_in_candidate:
                    continue

                if __start_node == __end_node:
                    continue

                if not graph.has_edge(__start_node, __end_node):
                    if self.weighted_graph:
                        graph.add_edge(__start_node, __end_node, weight=0.0)
                    else:
                        graph.add_edge(__start_node, __end_node)

                if self.weighted_graph:
                    # SingleRank employ weight as the co-occurrence times
                    graph[__start_node][__end_node]['weight'] += 1.0
        if self.page_rank_bias_type == 'position_rank':
            bias = {}
            for n, s in enumerate(stemmed_tokens):
                if s not in unique_tokens_in_candidate:
                    continue
                if s not in bias.keys():
                    bias[s] = 1 / (1 + n)
                else:
                    bias[s] += 1 / (1 + n)
            norm = sum(bias.values())
            bias = {k: v/norm for k, v in bias.items()}
        elif self.page_rank_bias_type == 'lda':
            # pagerank to get score for individual word (sum of score will be one)
            word_matrix = self.lda.probability_word()  # topic size x vocab
            topic_dist = self.lda.distribution_topic_document(document)
            topic_vector = np.array([i for t, i in topic_dist])  # topic size

            # single TPR
            vocab = self.lda.dictionary.token2id

            def norm_inner(a, b):
                return np.sum(a * b) / (np.sqrt(np.sum(a * a) + np.sum(b * b)) + 1e-7)

            bias = dict()
            for word in unique_tokens_in_candidate:
                if word in vocab.keys():
                    word_id = vocab[word]
                    word_vector = word_matrix[:, word_id]
                    bias[word] = norm_inner(word_vector, topic_vector)
                else:
                    bias[word] = 0.0

            # normalize the topical word importance of words
            norm = sum(bias.values())
            bias = {k: v / norm for k, v in bias.items()}
        elif self.page_rank_bias_type in ['tfidf', 'lex_spec']:
            if self.page_rank_bias_type == 'tfidf':
                dist_word = self.tfidf.distribution_word(document)
            else:
                dist_word = self.lexical.lexical_specificity(document)
            bias = {word: dist_word[word] if word in dist_word.keys() else 0.0 for word in unique_tokens_in_candidate}
            norm = sum(bias.values())
            bias = {k: v / norm for k, v in bias.items()}
        else:
            bias = None
        return graph, phrase_instance, len(stemmed_tokens), bias

    def run_pagerank(self, graph, personalization=None):
        if personalization:
            return nx.pagerank(
                G=graph, alpha=self.__random_prob, tol=self.__tol, weight='weight', personalization=personalization)
        else:
            return nx.pagerank_scipy(G=graph, alpha=self.__random_prob, tol=self.__tol, weight='weight')


class SingleRank(TextRank):
    """ SingleRank """

    def __init__(self, window_size: int = 2, *args, **kwargs):
        """ SingleRank """
        super(SingleRank, self).__init__(window_size=window_size, *args, **kwargs)
        self.weighted_graph = True


class PositionRank(TextRank):
    """ PositionRank """

    def __init__(self, window_size: int = 2, *args, **kwargs):
        """ PositionRank """
        super(PositionRank, self).__init__(window_size=window_size, *args, **kwargs)
        self.weighted_graph = True
        self.page_rank_bias_type = 'position_rank'


class TFIDFRank(TextRank):
    """ TFIDFRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ ExpandRank """
        super(TFIDFRank, self).__init__(language=language, *args, **kwargs)
        self.tfidf = TFIDF(language=language)
        self.weighted_graph = True
        self.prior_required = True
        self.page_rank_bias_type = 'tfidf'

    def load(self, directory: str = None):
        self.tfidf.load(directory)

    def train(self, data: list, export_directory: str = None):
        self.tfidf.train(data, export_directory=export_directory)


class LexRank(TextRank):
    """ LexRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ LexRank """
        super(LexRank, self).__init__(language=language, *args, **kwargs)
        self.lexical = LexSpec(language=language)
        self.weighted_graph = True
        self.prior_required = True
        self.page_rank_bias_type = 'lex_spec'

    def load(self, directory: str = None):
        self.lexical.load(directory)

    def train(self, data: list, export_directory: str = None):
        self.lexical.train(data, export_directory=export_directory)


class SingleTPR(TextRank):
    """ Single TopicalPageRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ Single TopicalPageRank """

        super(SingleTPR, self).__init__(language=language, *args, **kwargs)
        self.lda = LDA(language=language)
        self.directed_graph = True
        self.weighted_graph = True
        self.prior_required = True
        self.page_rank_bias_type = 'lda'

    def load(self, directory: str = None):
        self.lda.load(directory)

    def train(self, data: list, export_directory: str = None, num_topics: int = 15):
        self.lda.train(data, export_directory=export_directory, num_topics=num_topics)
