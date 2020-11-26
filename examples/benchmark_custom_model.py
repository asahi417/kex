""" Benchmark custom method """
import logging
import json
from time import time
from tqdm import tqdm

import grapher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging


class CustomExtractor:
    """ Custom keyword extractor example: First N keywords extractor """

    def __init__(self, maximum_word_number: int = 3):
        """ First N keywords extractor """
        self.phrase_constructor = grapher.PhraseConstructor(maximum_word_number=maximum_word_number)

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
        sorted_phrases = sorted(phrase_instance.values(), key=lambda x: x['offset'][0][0])
        return sorted_phrases[:min(len(sorted_phrases), n_keywords)]


if __name__ == '__main__':
    model = CustomExtractor()
    data_name = 'Inspec'

    data, _ = grapher.get_benchmark_dataset(data_name)

    # run algorithm and test it over data
    tp = {"5": 0, "10": 0, "15": 0}
    fn = {"5": 0, "10": 0, "15": 0}
    fp = {"5": 0, "10": 0, "15": 0}

    start = time()
    for v in tqdm(data):
        source = v['source']
        gold_keys = v['keywords']   # already stemmed
        # inference
        keys = model.get_keywords(source, n_keywords=15)
        keys_stemmed = [k['stemmed'] for k in keys]
        for i in [5, 10, 15]:
            positive_answers = list(set(keys_stemmed[:i]).intersection(set(gold_keys)))
            tp[str(i)] += len(positive_answers)
            fn[str(i)] += len(gold_keys) - len(positive_answers)
            fp[str(i)] += i - len(positive_answers)
    elapsed = time() - start
    # result summary: micro F1 score
    result_json = {'process_time_second': elapsed}
    for i in [5, 10, 15]:
        precision = tp[str(i)]/(fp[str(i)] + tp[str(i)])
        recall = tp[str(i)]/(fn[str(i)] + tp[str(i)])
        f1 = 2 * precision * recall / (precision + recall)
        result_json['top_{}'.format(i)] = {"mean_recall": recall, "mean_precision": precision, "f_1": f1}
    logging.info(json.dumps(result_json, indent=4, sort_keys=True))

