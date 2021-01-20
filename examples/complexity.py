""" Benchmark preset methods in grapher """
import argparse
import logging
import json
from tqdm import tqdm
from time import time

import grapher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
trial = 100
data, language = grapher.get_benchmark_dataset("Inspec")


def run_model(model_name: str):
    model = grapher.AutoAlgorithm(model_name, language=language)
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

    model_list = grapher.VALID_ALGORITHMS

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

    with open('{}/complexity.json'.format(export_dir_root), 'w') as f:
        json.dump(complexity, f)


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in grapher')
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    measure_complexity(opt.export)