""" Get statistics of grapher's build in dataset """
import argparse
import os
import logging
from tqdm import tqdm

import pandas as pd
import grapher
from glob import glob

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in grapher')
    parser.add_argument('-d', '--data', help='data:{}'.format(grapher.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    data_list = grapher.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
    stats = [
        'filename', 'n_phrase',
        'n_label', 'n_label_in_candidates', 'n_label_out_candidates', 'n_label_intractable',
        'n_word', 'n_unique_word', 'mean', 'std',
        'n_word_with_stopword', 'n_unique_word_with_stopword', 'mean_with_stopword', 'std_with_stopword',
        'label_in_candidates', 'label_out_candidates', 'label_intractable'
    ]
    target_stats = ["n_phrase", "n_label", "n_label_in_candidates", "n_label_out_candidates", "n_label_intractable",
                    "n_word", "n_unique_word", "n_word_with_stopword", "n_unique_word_with_stopword"]
    df_stats_all = pd.DataFrame(index=target_stats)
    all_stats = []
    for data_n, data in enumerate(data_list):
        logging.info('data: {}'.format(data))
        logging.info('computing stats on each data')
        _file = '{}/{}/statistics.csv'.format(opt.export, data)
        if os.path.exists(_file):
            logging.info(' - skip statistics computation: statistics file found at {}'.format(_file))
            df_stats = pd.read_csv(_file, index_col=0)
        else:
            os.makedirs(os.path.join(opt.export, data), exist_ok=True)
            df_stats = pd.DataFrame(index=stats)
            tmp, language = grapher.get_benchmark_dataset(data, keep_only_valid_label=False)
            for n, i in enumerate(tqdm(tmp)):
                keywords, source, filename = i['keywords'], i['source'], i['id']
                out = grapher.get_statistics(keywords, source)
                out['label_in_candidates'] = '||'.join(out['label_in_candidates'])
                out['label_out_candidates'] = '||'.join(out['label_out_candidates'])
                out['label_intractable'] = '||'.join(out['label_intractable'])
                out['filename'] = filename
                df_stats[n] = [out[i] for i in stats]
            df_stats = df_stats.T
            df_stats.to_csv(_file)

        df_stats_all[data] = [df_stats[i].mean() for i in target_stats]
    df_stats_all = df_stats_all.T
    df_stats_all.to_csv('{}/statistics_all.csv'.format(opt.export))
    for i in glob('{}/full-result.*.csv'.format(opt.export)):
        n = i.split('full-result.')[-1].replace('.csv', '')
        tmp = pd.read_csv(i, index_col=0)
        pd.merge(df_stats_all, tmp, right_index=True, left_index=True).to_csv(
            '{}/statistics_all_{}.csv'.format(opt.export, n))

