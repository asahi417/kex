""" Implementation of the LexRank """
from .lexical_specificity import LexicalSpec
from .graph_text_rank import TextRank


class LexRank(TextRank):
    """ LexRank """

    def __init__(self, language: str = 'en', *args, **kwargs):
        """ LexRank """
        super(LexRank, self).__init__(language=language, *args, **kwargs)
        self.lexical = LexicalSpec(language=language)
        self.directed_graph = False
        self.weighted_graph = True
        self.prior_required = True

    def load(self, directory: str = None):
        self.lexical.load(directory)

    def train(self, data: list, export_directory: str = None):
        self.lexical.train(data, export_directory=export_directory)

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

        ls_dict = self.lexical.lexical_specificity(document)
        bias = dict()
        for word in graph.nodes():
            if word in ls_dict.keys():
                bias[word] = ls_dict[word]
            else:
                bias[word] = 0.0

        node_score = self.run_pagerank(graph, personalization=bias)

        # combine score to get score of phrase
        phrase_score = [
            (candidate_phrase_stemmed_form, sum(node_score[t] for t in candidate_phrase_stemmed_form.split()))
            for candidate_phrase_stemmed_form in phrase_instance.keys()
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
