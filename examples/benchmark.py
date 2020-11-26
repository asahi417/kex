""" Benchmark
python examples/benchmark.py -m TopicRank,TextRank,PositionRank,SingleRank
"""
import argparse
import os
import logging
import json
from time import time
from glob import glob
from tqdm import tqdm

import pandas as pd
import grapher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging


def view_result(export_dir: str):
    d = 2
    all_data = list(set('.'.join(os.path.basename(i).split('.')[:-2]) for i in glob('{}/*.json'.format(export_dir))))
    all_algorithm = list(set(os.path.basename(i).split('.')[-2] for i in glob('{}/*.json'.format(export_dir))))
    df = {i: pd.DataFrame(index=all_data, columns=all_algorithm) for i in ['5', '10', '15', 'time']}
    for i in glob('{}/*.json'.format(export_dir)):
        algorithm = i.split('.')[-2]
        data_name = '.'.join(os.path.basename(i).split('.')[:-2])
        tmp = json.load(open(i))
        for n in ['5', '10', '15']:
            df[n][algorithm][data_name] = "{0} ({1}/{2})".format(
                round(tmp['top_{}'.format(n)]['f_1'] * 100, d),
                round(tmp['top_{}'.format(n)]['mean_precision'] * 100, d),
                round(tmp['top_{}'.format(n)]['mean_recall'] * 100, d))

        df['time'][algorithm][data_name] = str(round(tmp['process_time_second'], d))
    return df


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark algorithms on SemEval2010')
    parser.add_argument('-m', '--model', help='model:{}'.format(grapher.VALID_ALGORITHMS), default=None, type=str)
    parser.add_argument('-d', '--data', help='data:{}'.format(grapher.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    parser.add_argument('--view', help='view results', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    eps = 1e-6
    opt = get_options()
    if opt.view:
        _df = view_result(opt.export)
        for k, v in _df.items():
            logging.info("** Result: {} **\n {}".format(k, v))
            v.to_csv("{}/full-result.{}.csv".format(opt.export, k))
        exit()

    data_list = grapher.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
    algorithm_list = grapher.VALID_ALGORITHMS if opt.model is None else opt.model.split(',')
    for data_name in data_list:
        # load dataset
        data, language = grapher.get_benchmark_dataset(data_name)

        # load model
        for algorithm_name in algorithm_list:
            model = grapher.AutoAlgorithm(algorithm_name, language=language)
            logging.info('Benchmark')
            logging.info(' - algorithm: {}\n - data: {}'.format(algorithm_name, data_name))

            # compute prior
            if model.prior_required:
                logging.info(' - computing prior...')
                try:
                    model.load('./cache/{}/priors'.format(data_name))
                except Exception:
                    model.train([i['source'] for i in data], export_directory='./cache/{}/priors'.format(data_name))

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

            # export as a json
            os.makedirs(opt.export, exist_ok=True)
            export = os.path.join(opt.export, '{}.{}.json'.format(data_name.lower(), algorithm_name.lower()))
            with open(export, 'w') as f:
                json.dump(result_json, f)
            logging.info(' - result exported to {}'.format(export))
            logging.info(json.dumps(result_json, indent=4, sort_keys=True))

