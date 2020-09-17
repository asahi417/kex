""" Implementation of the ExpandRank """
from .tfidf import TFIDF
from .graph_text_rank import TextRank


class ExpandRank(TextRank):
    """ TopicalPageRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ TopicalPageRank """

        super(ExpandRank, self).__init__(language=language, *args, **kwargs)
        self.tfidf = TFIDF(language=language)
        self.directed_graph = False
        self.weighted_graph = True

    def save(self, directory: str = None):
        self.tfidf.save(directory)

    def load(self, directory: str = None):
        self.tfidf.load(directory)

    def train(self, data: list, export_directory: str = None, num_topics: int = 15):
        self.tfidf.train(data, export_directory=export_directory)

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

        # make graph and get data structure for candidate phrase
        graph, phrase_instance, original_sentence_token_size = self.build_graph(document)

        # pagerank to get score for individual word (sum of score will be one)
        dist_word = self.tfidf.distribution_word(document)
        bias = dict()
        for word in graph.nodes():
            if word in dist_word.keys():
                bias[word] = dist_word[word]
            else:
                bias[word] = 0.0

        node_score = self.run_pagerank(graph, personalization=bias)

        # combine score to get score of phrase
        phrase_score_dict = dict()
        for candidate_phrase_stemmed_form in phrase_instance.keys():
            tokens_in_phrase = candidate_phrase_stemmed_form.split()
            phrase_score_dict[candidate_phrase_stemmed_form] = sum([node_score[t] for t in tokens_in_phrase])

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
