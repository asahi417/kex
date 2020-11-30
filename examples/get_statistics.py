""" Benchmark preset methods in grapher """
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


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in grapher')
    parser.add_argument('-d', '--data', help='data:{}'.format(grapher.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    data_list = grapher.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
    stats = ['n_phrase', 'n_label', 'n_word', 'n_unique_word', 'mean', 'std',
             'n_word_with_stopword', 'n_unique_word_with_stopword', 'mean_with_stopword', 'std_with_stopword']
    for data_n, data in enumerate(data_list):
        if os.path.exists('{}/{}/statistics.csv'.format(opt.export, data)):
            continue
        logging.info('data: {}'.format(data))
        os.makedirs(os.path.join(opt.export, data), exist_ok=True)
        df_stats = pd.DataFrame(index=stats)
        tmp, language = grapher.get_benchmark_dataset(data)
        for n, i in enumerate(tqdm(tmp)):
            keywords, source = i['keywords'], i['source']
            out = grapher.get_statistics(keywords, source)
            df_stats[n] = [out[i] for i in stats]
        df_stats.T.to_csv('{}/{}/statistics.csv'.format(opt.export, data))
