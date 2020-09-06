""" TFIDF based keyphrase extraction algorithm.
raw document -> list of phrase := [p1, .. pn], sum up each token's tfidf prob in each phrase

If `data_name` is false -> just term frequency model
"""

import numpy as np
from ..tfidf import TFIDF
from ..processing import Processing


class TFIDFBased:

    def __init__(self,
                 language: str = 'en',
                 tol: float = 0.0001,
                 data_name: str = None,
                 add_verb: bool = False):
        """ TextRank algorithm

         Parameter
        ------------------------
        language: str
            `en` or `jp`
        tol: float
            PageRank parameter that define tolerance of convergence
        data_name: str
            (development) load pre-computed tfidf matrix
        add_verb: bool
            take the verb into account
        """

        self.__tol = tol
        self.__data_name = data_name
        self.tfidf = TFIDF(debug=False)
        if self.__data_name is not None:
            self.tfidf.load(self.__data_name)
            self.global_tfidf = True
        else:
            self.global_tfidf = False
        self.processor = Processing(language=language, add_verb=add_verb)
        self.list_of_phrase = []

    def extract(self,
                target_documents: list,
                count: int = 10):
        """ Extract keyphrases from list of documents

         Parameter
        ------------------------
        target_documents: list
            list of document
        count: int
            number of keyphrase to extract

         Return
        ------------------------
        list of keyphrase and related information extracted from each target document
        """

        if len(target_documents) < 0:
            return []

        output = [self.processor(t, return_token=True) for t in target_documents]

        def get_phrase_single_document(phrase_instance, cleaned_document_tokenized):
            # convert phrase instance
            self.list_of_phrase.append(phrase_instance)

            def aggregate_prob(__phrase_key):
                __phrase = phrase_instance[__phrase_key]

                aggregation = np.mean

                if not self.global_tfidf:
                    prob = __phrase['count']
                else:
                    id_prob = self.tfidf.distribution_word(__phrase.split(), return_word_id=True)
                    prob = float(aggregation([_prob for _id, _prob in id_prob]))
                return prob

            phrase_prob = [(k, aggregate_prob(k)) for k in phrase_instance.keys()]

            # sorting
            phrase_score_sorted_list = sorted(phrase_prob, key=lambda id_score: id_score[1], reverse=True)
            count_valid = min(count, len(phrase_score_sorted_list))
            original_sentence_token_size = len(cleaned_document_tokenized)

            def modify_output(stem, score):
                tmp = phrase_instance[stem]
                tmp['score'] = score
                tmp['original_sentence_token_size'] = original_sentence_token_size
                return tmp

            val = [modify_output(stem, score) for stem, score in phrase_score_sorted_list[:count_valid]]

            return val

        values = [get_phrase_single_document(out[0], out[1]) if out else [] for out in output]

        return values
