""" Benchmark preset methods in kex """
import argparse
import logging
import json
import os
from tqdm import tqdm
from time import time

import kex
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
trial = 100
data, language = kex.get_benchmark_dataset("Inspec")


def run_model(model_name: str):
    model = kex.get_algorithm(model_name, language=language)
    elapse_prior = None
    if model.prior_required:
        try:
            model.load('./cache/complexity')
        except Exception:
            start = time()
            model.train([i['source'] for i in data], export_directory='./cache/complexity')
            elapse_prior = time() - start
    start = time()
    for v in tqdm(data):
        model.get_keywords(v['source'])
    elapse = time() - start
    return elapse, elapse_prior


def measure_complexity(export_dir_root: str = './benchmark'):
    """ Run keyword extraction benchmark """
    _file = '{}/complexity.json'.format(export_dir_root)
    if os.path.exists(_file):
        with open(_file, 'r') as f:
            complexity = json.load(f)
    else:
        model_list = kex.VALID_ALGORITHMS

        logging.info('Measure complexity')
        complexity = {}
        for model_name in model_list:
            logging.info(' - algorithm: {}'.format(model_name))
            complexity[model_name] = {}
            n = 0
            elapse_list = []
            elapse_prior_list = []
            while n < trial:
                elapse, elapse_prior = run_model(model_name)
                elapse_list.append(elapse)
                if elapse_prior is not None:
                    elapse_prior_list.append(elapse_prior)
                n += 1
            complexity[model_name]['elapse'] = sum(elapse_list) / len(elapse_list)
            if len(elapse_prior_list) > 0:
                complexity[model_name]['elapse_prior'] = sum(elapse_prior_list) / len(elapse_prior_list)
            else:
                complexity[model_name]['elapse_prior'] = 0

        with open(_file, 'w') as f:
            json.dump(complexity, f)

    df = pd.DataFrame(complexity).T
    df_tmp = df['elapse'].round(1)
    df_tmp.name = 'Time (sec.)'
    df_tmp.to_csv('{}/complexity.csv'.format(export_dir_root))
    pd.DataFrame({
        'TF': {'Model': 'TF,LexSpec,LexRank', 'Time (sec.)': df['elapse_prior']['TF'].round(1)},
        'TFIDF': {'Model': 'TFIDF,TFIDFRank', 'Time (sec.)': df['elapse_prior']['TFIDF'].round(1)},
        'LDA': {'Model': 'SingleTPR', 'Time (sec.)': df['elapse_prior']['SingleTPR'].round(1)}
    }).to_csv('{}/complexity.prior.csv'.format(export_dir_root))


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in kex')
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    measure_complexity(opt.export)
