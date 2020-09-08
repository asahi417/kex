""" Benchmark algorithms on SemEval2010
* As described in the paper (https://www.aclweb.org/anthology/S10-1004.pdf), F1 score on top 5/10/15 F1 score is
employed.
"""
import argparse
import os
import logging
import json
from time import time
from logging.config import dictConfig

import grapher
from tqdm import tqdm

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark algorithms on SemEval2010')
    parser.add_argument('-m', '--model', help='model', default='TopicRank', type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


def mean(_list):
    return sum(_list)/len(_list)


if __name__ == '__main__':
    opt = get_options()

    # load model
    lda = False
    tfidf = False

    if opt.model == 'TopicRank':
        model = grapher.TopicRank()
    elif opt.model == 'TextRank':
        model = grapher.TextRank()
    elif opt.model == 'SingleRank':
        model = grapher.SingleRank()
    elif opt.model == 'MultipartiteRank':
        model = grapher.MultipartiteRank()
    elif opt.model == 'PositionRank':
        model = grapher.PositionRank()
    elif opt.model == 'TFIDF':
        model = grapher.TFIDF()
        tfidf = True
    elif opt.model == 'TopicalPageRank':
        model = grapher.TopicalPageRank()
        lda = True
    elif opt.model == 'SingleTopicalPageRank':
        model = grapher.SingleTopicalPageRank()
        lda = True
    else:
        raise ValueError('unknown model: {}'.format(opt.model))
    LOGGER.info('Benchmark on SemEval2010 keyphrase extraction dataset')
    LOGGER.info('algorithm: {}'.format(opt.model))

    # load dataset
    data = grapher.get_benchmark_dataset('SemEval2010')

    # compute prior
    if lda:
        LOGGER.info('computing LDA prior...')
        try:
            model.load('./cache/lda_semeval')
        except Exception:
            corpus = [i['source'] for i in data]
            model.train(corpus, export_directory='./cache/lda_semeval')
    if tfidf:
        LOGGER.info('computing TFIDF prior...')
        try:
            model.load('./cache/tfidf_semeval')
        except Exception:
            corpus = [i['source'] for i in data]
            model.train(corpus, export_directory='./cache/tfidf_semeval')

    # run algorithm and test it over data
    precisions = {}
    recalls = {}
    start = time()
    for v in tqdm(data):
        source = v['source']
        gold_keys = v['keywords']   # already stemmed

        # inference
        keys = model.get_keywords(source, n_keywords=15)
        keys_stemmed = [k['stemmed'] for k in keys]

        # get an intersection
        for i in [5, 10, 15]:
            positive_answers = list(set(keys_stemmed[:i]).intersection(set(gold_keys)))
            precisions[str(i)] = [len(positive_answers) / i]
            recalls[str(i)] = [len(positive_answers) / len(gold_keys)]
    elapsed = time() - start

    # result summary
    result_json = {'process_time_second': elapsed}
    for i in [5, 10, 15]:
        ave_recall = mean(recalls[str(i)])
        ave_precision = mean(precisions[str(i)])
        result_json['top_{}'.format(i)] = {
            "mean_recall": ave_recall,
            "mean_precision": ave_precision,
            "f_1": 2*ave_precision*ave_recall/(ave_precision+ave_recall)
        }
    # export as a json
    os.makedirs(opt.export, exist_ok=True)
    with open(os.path.join(opt.export, 'semeval.{}.json'.format(opt.model.lower())), 'w') as f:
        json.dump(result_json, f)
    LOGGER.info('result exported to {}'.format(os.path.join(opt.export, 'semeval.{}.json'.format(opt.model.lower()))))
    LOGGER.info(json.dumps(result_json, indent=4, sort_keys=True))
