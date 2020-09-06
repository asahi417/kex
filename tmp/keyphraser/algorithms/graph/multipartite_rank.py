"""MultipartiteRank keyphrase extraction model.
Implementation of the MultipartiteRank model for keyword extraction.
"""

import networkx as nx
import numpy as np
from itertools import chain
from .topic_rank import TopicRank


class MultipartiteRank(TopicRank):
    """MultipartiteRank

     Usage
    -----------------
    >>> model = MultipartiteRank(window_size=10)
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
                 parameter_adjust_weight: float = 1.1,
                 random_prob: float = 0.85,
                 tol: float = 0.0001,
                 clustering_threshold: float = 0.74,
                 linkage_method: str = 'average'
                 ):
        """ TopicRank algorithm

        :param stop_words: Additional stopwords (As default, NLTK English stopwords is contained)
        :param random_prob: PageRank parameter that is coefficient of convex combination of random suffer model
        :param clustering_threshold: parameter for clustering (default should is the one used in original paper)
        :param linkage_method: parameter for clustering (default should is the one used in original paper)
        :param tol: PageRank parameter that define tolerance of convergence
        """

        super(MultipartiteRank, self).__init__(language=language,
                                               random_prob=random_prob,
                                               tol=tol,
                                               clustering_threshold=clustering_threshold,
                                               linkage_method=linkage_method)
        self.parameter_adjust_weight = parameter_adjust_weight
        self.directed_graph = True
        self.weighted_graph = True

    def build_graph(self, target_document: str):
        """ build graph with topics """

        # convert phrase instance
        phrase_instance, cleaned_document_tokenized = self.processor(target_document, return_token=True)

        # group phrases as topic classes
        grouped_phrases = self.topic_clustering(list(phrase_instance.keys()))
        group_id = list(range(len(grouped_phrases)))

        def get_group_id(__phrase):
            return np.argmax([int(__phrase in g) for g in grouped_phrases])

        # initialize graph instance
        graph = nx.DiGraph()

        # add nodes
        unique_tokens_in_candidate = list(set(chain(*[i.split() for i in phrase_instance.keys()])))
        graph.add_nodes_from(unique_tokens_in_candidate)

        # add edges
        for position, __start_node in enumerate(cleaned_document_tokenized):

            # ignore invalid token
            if __start_node not in unique_tokens_in_candidate:
                continue

            for __position in range(position, len(cleaned_document_tokenized)):
                __end_node = cleaned_document_tokenized[__position]

                # ignore invalid token
                if __end_node not in unique_tokens_in_candidate:
                    continue

                # ignore intra-topic connection
                if get_group_id(__start_node) == get_group_id(__end_node):
                    continue

                weight = 1 / abs(position - __position)

                if not graph.has_edge(__start_node, __end_node):
                    graph.add_edge(__start_node, __end_node, weight=weight)
                    graph.add_edge(__end_node, __start_node, weight=weight)
                else:
                    graph[__start_node][__end_node]['weight'] += weight
                    graph[__end_node][__start_node]['weight'] += weight

        # weight adjustment
        for i in group_id:
            stemmed_phrases = grouped_phrases[i]
            offset_phrase = [(phrase_instance[phrase]['offset'][0][0], phrase) for phrase in stemmed_phrases]
            offset_phrase_sort = sorted(offset_phrase, key=lambda key_value: key_value[0])
            first_position, first_phrase = offset_phrase_sort[0]
            scale = self.parameter_adjust_weight * np.exp(1 / (1+first_position))

            neighbour_w = [
                (
                    np.sum([
                        graph[__start][__end]['weight']
                        for __start, __end in graph.edges(start)
                        if get_group_id(__start) == i and __start != first_phrase and __end == start
                    ]),
                    start
                )
                for start, end in graph.edges(first_phrase) if end == first_phrase
            ]
            for w, start in neighbour_w:
                graph[start][first_phrase]['weight'] += scale*w

        original_sentence_token_size = len(cleaned_document_tokenized)
        return graph, phrase_instance, grouped_phrases, original_sentence_token_size

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
