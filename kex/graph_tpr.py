""" Implementation of the TopicalPageRank """
import numpy as np
from .lda import LDA
from .graph_text_rank import TextRank


class TopicalPageRank(TextRank):
    """ TopicalPageRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ TopicalPageRank """

        super(TopicalPageRank, self).__init__(language=language, *args, **kwargs)
        self.lda = LDA(language=language)
        self.directed_graph = True
        self.weighted_graph = True
        self.prior_required = True

    def load(self, directory: str = None):
        self.lda.load(directory)

    def train(self, data: list, export_directory: str = None, num_topics: int = 15):
        self.lda.train(data, export_directory=export_directory, num_topics=num_topics)

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

        # make graph and get data structure for candidate phrase
        graph, phrase_instance, original_sentence_token_size = self.build_graph(document)

        # pagerank to get score for individual word (sum of score will be one)
        word_matrix = self.lda.probability_word()  # topic size x vocab
        topic_dist = self.lda.distribution_topic_document(document)
        topic_vector = np.array([i for t, i in topic_dist])  # topic size

        # original TPR procedure
        __node_score = []

        for i in range(self.lda.topic_size):
            bias = dict()
            vocab = self.lda.dictionary.token2id
            unk = 0
            for word in graph.nodes():
                if word in vocab.keys():
                    word_id = vocab[word]
                    bias[word] = word_matrix[i, word_id]
                else:
                    bias[word] = 0.0
                    unk += 1

            # normalize the topical word importance of words
            norm = sum(bias.values())
            for word in bias:
                if norm == 0:
                    bias[word] = 1 / len(bias[word])
                else:
                    bias[word] /= norm
            __node_score.append(self.run_pagerank(graph, personalization=bias))

        # combine score to get score of phrase
        phrase_score_dict = [
            (candidate_phrase_stemmed_form, sum(
                sum(__node_score[i][t] for t in candidate_phrase_stemmed_form.split()) * topic_vector[i]
                for i in range(self.lda.topic_size)))
            for candidate_phrase_stemmed_form in phrase_instance.keys()
        ]

        # sorting
        phrase_score_sorted_list = sorted(phrase_score_dict, key=lambda key_value: key_value[1], reverse=True)
        count_valid = min(n_keywords, len(phrase_score_sorted_list))

        def modify_output(stem, _score):
            tmp = phrase_instance[stem]
            tmp['score'] = _score
            tmp['n_source_tokens'] = original_sentence_token_size
            return tmp

        val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]
        return val
