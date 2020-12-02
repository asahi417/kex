# """ Implementation of MultipartiteRank """
# import networkx as nx
# import numpy as np
# from itertools import chain
# from .graph_topic_rank import TopicRank
#
# __all__ = "MultipartiteRank"
#
#
# class MultipartiteRank(TopicRank):
#     """ MultipartiteRank
#
#      Usage
#     -----------------
#     >>> model = MultipartiteRank()
#     >>> sample =
#     'We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier '
#     'detection. It starts by training word embeddings on the target document to capture semantic regularities among '
#     'the words. It then uses the minimum covariance determinant estimator to model the distribution of non-keyphrase '
#     'word vectors, under the assumption that these vectors come from the same distribution, indicative of their '
#     'irrelevance to the semantics expressed by the dimensions of the learned vector representation. Candidate '
#     'keyphrases only consist of words that are detected as outliers of this dominant distribution. Empirical results '
#     'show that our approach outperforms stateof-the-art and recent unsupervised keyphrase extraction methods.'
#     >>> model.get_keywords(sample)
#     [[
#         ('novel unsupervis keyphras extract approach', 0.2570445598381217,
#            {'stemmed': 'novel unsupervis keyphras extract approach',
#             'pos': 'ADJ ADJ NOUN NOUN NOUN',
#             'raw': ['novel unsupervised keyphrase extraction approach'],
#             'lemma': ['novel unsupervised keyphrase extraction approach'],
#             'offset': [[3, 7]],
#             'count': 1}),
#         ('keyphras word vector', 0.19388916977478182,
#            {'stemmed': 'keyphras word vector',
#             'pos': 'NOUN NOUN NOUN',
#             'raw': ['keyphrase word vectors'],
#             'lemma': ['keyphrase word vector'],
#             'offset': [[49, 51]],
#             'count': 1})
#     ]]
#     """
#
#     def __init__(self, parameter_adjust_weight: float = 1.1, *args, **kwargs):
#         """ MultipartiteRank """
#
#         super(MultipartiteRank, self).__init__(*args, **kwargs)
#         self.parameter_adjust_weight = parameter_adjust_weight
#         self.weighted_graph = True
#
#     def build_graph(self, document: str):
#         """ Build basic graph with Topic """
#
#         # convert phrase instance
#         phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
#         if len(phrase_instance) < 2:
#             # at least 2 phrase are needed to extract keywords
#             return None
#
#         # group phrases as topic classes
#         grouped_phrases = self.topic_clustering(list(phrase_instance.keys()))
#         group_id = list(range(len(grouped_phrases)))
#
#         # initialize graph instance
#         graph = nx.DiGraph()
#
#         # add nodes
#         unique_tokens_in_candidate = list(set(chain(*[i.split() for i in phrase_instance.keys()])))
#         graph.add_nodes_from(unique_tokens_in_candidate)
#
#         def get_group_id(__phrase):
#             return np.argmax([int(__phrase in g) for g in grouped_phrases])
#
#         # add edges
#         for position, __start_node in enumerate(stemmed_tokens):
#
#             # ignore invalid token
#             if __start_node not in unique_tokens_in_candidate:
#                 continue
#
#             for __position in range(position, len(stemmed_tokens)):
#                 __end_node = stemmed_tokens[__position]
#
#                 # ignore invalid token
#                 if __end_node not in unique_tokens_in_candidate:
#                     continue
#
#                 # ignore intra-topic connection
#                 if get_group_id(__start_node) == get_group_id(__end_node):
#                     continue
#
#                 weight = 1 / abs(position - __position)
#
#                 if not graph.has_edge(__start_node, __end_node):
#                     graph.add_edge(__start_node, __end_node, weight=weight)
#                     graph.add_edge(__end_node, __start_node, weight=weight)
#                 else:
#                     graph[__start_node][__end_node]['weight'] += weight
#                     graph[__end_node][__start_node]['weight'] += weight
#
#         # weight adjustment
#         for i in group_id:
#             stemmed_phrases = grouped_phrases[i]
#             offset_phrase = [(phrase_instance[phrase]['offset'][0][0], phrase) for phrase in stemmed_phrases]
#             offset_phrase_sort = sorted(offset_phrase, key=lambda key_value: key_value[0])
#             first_position, first_phrase = offset_phrase_sort[0]
#             scale = self.parameter_adjust_weight * np.exp(1 / (1+first_position))
#
#             neighbour_w = [
#                 (
#                     np.sum([
#                         graph[__start][__end]['weight']
#                         for __start, __end in graph.edges(start)
#                         if get_group_id(__start) == i and __start != first_phrase and __end == start
#                     ]),
#                     start
#                 )
#                 for start, end in graph.edges(first_phrase) if end == first_phrase
#             ]
#             for w, start in neighbour_w:
#                 graph[start][first_phrase]['weight'] += scale*w
#
#         return graph, phrase_instance, grouped_phrases, len(stemmed_tokens)
#
#     def get_keywords(self, document: str, n_keywords: int = 10):
#         """ Get keywords
#
#         Parameter
#         ------------------
#         document: str
#         n_keywords: int
#
#          Return
#         ------------------
#         a list of keyphrase with score eg) [('aa', 0.5), ('b', 0.3), ..]
#         """
#         # make graph and get data structure for candidate phrase
#         output = self.build_graph(document)
#         if output is None:
#             return []
#
#         graph, phrase_instance, grouped_phrases, original_sentence_token_size = output
#
#         # pagerank to get score for individual word (sum of score will be one)
#         node_score = self.run_pagerank(graph)
#
#         # combine score to get score of phrase
#         phrase_score = [
#             (candidate_phrase_stemmed_form, sum(node_score[t] for t in candidate_phrase_stemmed_form.split()))
#             for candidate_phrase_stemmed_form in phrase_instance.keys()
#         ]
#
#         # sorting
#         phrase_score_sorted_list = sorted(phrase_score, key=lambda key_value: key_value[1], reverse=True)
#         count_valid = min(n_keywords, len(phrase_score_sorted_list))
#
#         def modify_output(stem, score):
#             tmp = phrase_instance[stem]
#             tmp['score'] = score
#             tmp['n_source_tokens'] = original_sentence_token_size
#             return tmp
#
#         val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
#         return val
