""" Wrapper for algorithm instance to perform multi-docs keyphrase extraction """

import numpy as np
from itertools import chain

# how to use sentence length as weight for each keyphrase score
METHOD = ['linear', 'log', 'exp']


class MultiDocs:
    """ Keyphrase extraction for multiple document

     Usage
    --------------
    >>> from keyphraser import MultiDocs, algorithms
    >>> algorithms_instance = algorithms.graph.SingleRank()
    >>> multi_keyphraser = MultiDocs(algorithms_instance)

    """

    def __init__(self,
                 algorithm_instance,
                 method: str = 'linear'
                 # min_document_frequency: int=None,
                 # max_document_frequency: int = None
                 ):
        """ Multi docs keyphraser wrapper

         Parameter
        ------------------
        algorithm_instance:
            algorithm instance
        method: str
             method of aggregation. determine how to compute the score according to token size
        """

        self.__algorithm_instance = algorithm_instance
        if method not in METHOD:
            raise ValueError('unknown method: %s not in %s' % (method, METHOD))
        self.__method = method
        # self.__min_document_frequency = min_document_frequency
        # self.__max_document_frequency = max_document_frequency

    def extract(self,
                target_documents: list,
                count: int = 10):
        """ Extract keyphrase over multiple document.

         Parameter
        --------------
        target_documents: list
            target document
        count: int
            number of keyphrase to get

         Return
        -------------
        output: list of dict
             keyphrases for entire documents, sorted by score. each element is dictionary with key
            'raw', 'stemmed', 'score'.
        """

        output = self.__algorithm_instance.extract(target_documents)
        output = self.aggregation(output)
        output = output[:min(count, len(output))]
        return output

    def aggregation(self,
                    result_from_algorithm: list):
        """ Aggregate document-wise result to get keyphrases for entire documents

         Parameter
        --------------
        result_from_algorithm: list of dict
            result from document wise algorithm

         Return
        -------------
        grouped_output: list of dict
            keyphrases for entire documents, sorted by score. each element is dictionary with key
            'raw', 'stemmed', 'score'
        """
        if self.__method == 'linear':

            def __aggregation(_input):
                return _input
        elif self.__method == 'log':

            def __aggregation(_input):
                return np.log(_input + 1e-6)
        elif self.__method == 'exp':

            def __aggregation(_input):
                return np.exp(_input)
        else:
            raise ValueError('unknown method: %s not in %s' % (self.__method, METHOD))

        result_from_algorithm_concat = list(chain(*result_from_algorithm))
        document_wise_phrase = [[ii['stemmed'] for ii in i] for i in result_from_algorithm]
        unique_stemmed = set([r['stemmed'] for r in result_from_algorithm_concat])

        document_frequency = dict(
            [(s, np.sum([s in d for d in document_wise_phrase])) for s in unique_stemmed]
        )

        raw_score = [
            (r['raw'][0], r['stemmed'], float(r['score']) * __aggregation(r['original_sentence_token_size']))
            for r in result_from_algorithm_concat]

        # combine tokens shared same stemmed form (sums up score)
        def modify_output(output):
            modified_out = dict(raw=output[0][0],
                                stemmed=output[0][1],
                                score=sum([o[2] for o in output]),
                                document_frequency=document_frequency[output[0][1]])
            return modified_out

        grouped_output = [
            modify_output(list(filter(lambda x: x[1] == s, raw_score)))
            for s in unique_stemmed]

        grouped_output = sorted(grouped_output, key=lambda x: x['score'], reverse=True)
        return grouped_output

    @property
    def base_algorithm(self):
        """ base algorithm instance, which used to extract single document keyphrase """
        return self.__algorithm_instance
