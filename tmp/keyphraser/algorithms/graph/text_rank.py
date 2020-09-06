"""TextRank keyphrase extraction model.

Implementation of the TextRank (and its variants) model for keyword extraction described in:

* TextRank: Undirected Unweighted Graph
* SingleRank: Undirected Weighted Graph (weight is co-occurrence)
"""

import networkx as nx
from itertools import chain
from ..processing import Processing


class TextRank:
    """TextRank and its variants for keyword extraction.

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
        {
        'stemmed': 'novel unsupervis keyphras extract approach',
        'pos': 'ADJ ADJ NOUN NOUN NOUN',
        'raw': ['novel unsupervised keyphrase extraction approach'],
        'lemma': ['novel unsupervised keyphrase extraction approach'],
        'offset': [[3, 7]],
        'count': 1,
        'score': 0.2570445598381217,
        'original_sentence_token_size': 5
        },
        {
        'stemmed': 'keyphras word vector',
        'pos': 'NOUN NOUN NOUN',
        'raw': ['keyphrase word vectors'],
        'lemma': ['keyphrase word vector'],
        'offset': [[49, 51]],
        'count': 1,
        'score': 0.19388916977478182
        }
    ]]
    >>> model.list_of_graph  # check graphs
    """

    def __init__(self,
                 language: str='en',
                 window_size: int=10,
                 random_prob: float=0.85,
                 tol: float=0.0001,
                 add_verb: bool = False):
        """ TextRank algorithm

        :param window_size: Length of window to make edges
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param tol: PageRank parameter that define tolerance of convergence
        """

        self.__window_size = window_size
        self.__random_prob = random_prob
        self.__tol = tol

        self.processor = Processing(language=language, add_verb=add_verb)
        self.list_of_graph = []
        self.list_of_phrase = []
        self.directed_graph = False
        self.weighted_graph = False

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
        values = [self.get_phrase_single_document(doc, count=count) if len(str(doc)) > 2 and type(doc) is str else []
                  for doc in target_documents]
        return values

    def build_graph(self,
                    target_document: str,
                    directed_graph: bool = False,
                    weighted_graph: bool = False,
                    return_cleaned_document: bool = False):
        """ Build basic graph
        - nodes: phrases extracted from given document
        - edge: co-occurrence in certain window
        - weight: count of co-occurrence

        :param target_document:
        :param directed_graph:
        :param weighted_graph: bool=False
        :param return_cleaned_document: return cleaned tokenized document if True
        :return: graph instance, data structure of extracted phrase
        """

        # convert phrase instance
        phrase_instance, cleaned_document_tokenized = self.processor(target_document, return_token=True)
        if len(phrase_instance) < 2:
            # at least 2 phrase are needed to extract keyphrase
            return None

        # initialize graph instance
        if directed_graph:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # add nodes
        unique_tokens_in_candidate = list(set(chain(*[i.split() for i in phrase_instance.keys()])))
        graph.add_nodes_from(unique_tokens_in_candidate)

        # add edges
        for position, __start_node in enumerate(cleaned_document_tokenized[:-self.__window_size]):

            # ignore invalid token
            if __start_node not in unique_tokens_in_candidate:
                continue

            for __position in range(position, position + self.__window_size):
                __end_node = cleaned_document_tokenized[__position]

                # ignore invalid token
                if __end_node not in unique_tokens_in_candidate:
                    continue

                if not graph.has_edge(__start_node, __end_node):
                    graph.add_edge(__start_node, __end_node, weight=1.0)
                else:
                    if weighted_graph:
                        # SingleRank, ExpandRank employ weight as the co-occurrence times
                        graph[__start_node][__end_node]['weight'] += 1.0

        original_sentence_token_size = len(cleaned_document_tokenized)
        if return_cleaned_document:
            return graph, phrase_instance, original_sentence_token_size, cleaned_document_tokenized
        else:
            return graph, phrase_instance, original_sentence_token_size

    def run_pagerank(self,
                     graph,
                     personalization=None):
        """ Run PageRank to get score for each node

        :param graph:
        :param personalization: for biased PageRank
        :return: score for each node
        """
        if personalization:
            node_score = nx.pagerank(G=graph,
                                     alpha=self.__random_prob,
                                     tol=self.__tol,
                                     weight='weight',
                                     personalization=personalization)
        else:
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

        graph, phrase_instance, original_sentence_token_size = output

        self.list_of_graph.append(graph)
        self.list_of_phrase.append(phrase_instance)

        # pagerank to get score for individual word (sum of score will be one)
        node_score = self.run_pagerank(graph)

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
