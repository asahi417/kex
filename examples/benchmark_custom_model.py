""" Get top 10 micro F1 score for a custom method with Inspec dataset """
import logging
import json
from random import shuffle
from tqdm import tqdm

import kex

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging


class CustomExtractor:
    """ Custom keyword extractor example: random extractor """

    def __init__(self, maximum_word_number: int = 3):
        """ random keywords extractor """
        self.phrase_constructor = kex.PhraseConstructor(maximum_word_number=maximum_word_number)

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
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        phrase_list = list(phrase_instance.values())
        shuffle(phrase_list)
        return phrase_list[:min(len(phrase_list), n_keywords)]


if __name__ == '__main__':
    model = CustomExtractor()
    data, _ = kex.get_benchmark_dataset('Inspec')

    # run algorithm and test it over data
    tp, fn, fp = 0, 0, 0

    topn = 5
    # predictions = []
    # labels = []
    # for v in tqdm(data):
    #     source = v['source']
    #     gold_keys = v['keywords']   # already stemmed
    #     # inference
    #     predictions.append(model.get_keywords(source, n_keywords=topn))
    #     labels.append(gold_keys)

    for v in tqdm(data):
        source = v['source']
        gold_keys = v['keywords']   # already stemmed
        # inference
        keys = model.get_keywords(source, n_keywords=topn)
        keys_stemmed = [k['stemmed'] for k in keys]
        positive_answers = list(set(keys_stemmed[:topn]).intersection(set(gold_keys)))
        tp += len(positive_answers)
        fn += len(gold_keys) - len(positive_answers)
        fp += topn - len(positive_answers)

    # result summary: micro F1 score
    precision = tp/(fp + tp)
    recall = tp/(fn + tp)
    f1 = 2 * precision * recall / (precision + recall)
    logging.info(json.dumps({"precision": precision, "recall": recall, "f1": f1}, indent=4, sort_keys=True))

