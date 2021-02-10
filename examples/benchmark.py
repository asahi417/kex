""" Get top 10 micro F1 score and MRR """
import logging
import json
from tqdm import tqdm

import kex

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    model = kex.SingleRank()
    data, _ = kex.get_benchmark_dataset('Inspec')

    # run algorithm and test it over data
    tp, fn, fp = 0, 0, 0
    mrr = []
    topn = 10

    for v in tqdm(data):
        source = v['source']
        gold_keys = v['keywords']   # already stemmed
        # inference
        keys = model.get_keywords(source, n_keywords=1000)  # keep it higher for MRR computation
        keys_stemmed = [k['stemmed'] for k in keys]
        # metric
        positive_answers = list(set(keys_stemmed[:topn]).intersection(set(gold_keys)))
        all_ranks = [n + 1 for n, p_ in enumerate(keys_stemmed) if p_ in gold_keys]
        rank = min(all_ranks) if len(all_ranks) != 0 else len(keys_stemmed) + 1
        mrr.append(1 / rank)
        tp += len(positive_answers)
        fn += len(gold_keys) - len(positive_answers)
        fp += min(topn, len(gold_keys)) - len(positive_answers)

    # result summary: micro F1 score
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * precision * recall / (precision + recall)
    logging.info(json.dumps({
        "mrr": sum(mrr) / len(mrr),
        "precision": precision,
        "recall": recall,
        "f1": f1}, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
