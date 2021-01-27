""" Get statistics of kex's build in dataset """
import argparse
import os
import logging
from tqdm import tqdm
from itertools import chain

import pandas as pd
import kex


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging

types = {
    "Inspec": "Abst",
    "www": "Abst",
    "kdd": "Abst",
    "Krapivin2009": "Full",
    "SemEval2010": "Full",
    "SemEval2017": "Para",
    "citeulike180": "Full",
    "PubMed": "Full",
    "Schutz2008": "Full",
    "theses100": "Full",
    "fao30": "Full",
    "fao780": "Full",
    "Nguyen2007": "Full",
    "wiki20": "Report",
    "500N-KPCrowd-v1.1": "News"
}
domain = {
    "Inspec": "CS",
    "www": "CS",
    "kdd": "CS",
    "Krapivin2009": "CS",
    "SemEval2010": "CS",
    "SemEval2017": "-",
    "citeulike180": "BI",
    "PubMed": "BM",
    "Schutz2008": "BM",
    "theses100": "-",
    "fao30": "AG",
    "fao780": "AG",
    "Nguyen2007": "-",
    "wiki20": "CS",
    "500N-KPCrowd-v1.1": "-"
}
# BI: bioinfomatics
# BM biomedical
# AD: agricultural document

def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in kex')
    parser.add_argument('-d', '--data', help='data:{}'.format(kex.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    data_list = kex.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
    total_dict = {}
    for data_n, data in enumerate(data_list):
        logging.info('data: {}'.format(data))
        logging.info('computing stats on each data')
        _file = '{}/{}/statistics.csv'.format(opt.export, data)
        if os.path.exists(_file):
            logging.info(' - skip statistics computation: statistics file found at {}'.format(_file))
            df_stats = pd.read_csv(_file, index_col=0)
        else:
            stats = [
                'filename', 'n_phrase',
                'n_label', 'n_label_in_candidates', 'n_label_out_candidates', 'n_label_intractable',
                'n_word', 'n_unique_word', 'mean', 'std',
                'n_word_with_stopword', 'n_unique_word_with_stopword', 'mean_with_stopword', 'std_with_stopword',
                'label_in_candidates', 'label_out_candidates', 'label_intractable'
            ]
            os.makedirs(os.path.join(opt.export, data), exist_ok=True)
            df_stats = pd.DataFrame(index=stats)
            tmp, language = kex.get_benchmark_dataset(data, keep_only_valid_label=False)
            for n, i in enumerate(tqdm(tmp)):
                keywords, source, filename = i['keywords'], i['source'], i['id']
                out = kex.get_statistics(keywords, source)
                out['label_in_candidates'] = '||'.join(out['label_in_candidates'])
                out['label_out_candidates'] = '||'.join(out['label_out_candidates'])
                out['label_intractable'] = '||'.join(out['label_intractable'])
                out['filename'] = filename
                df_stats[n] = [out[i] for i in stats]
            df_stats = df_stats.T
            df_stats.to_csv(_file)

        tmp = len(list(chain(*[
            [k for k in i.split('||') if len(k.split(' ')) > 1]
            for i in df_stats['label_in_candidates'].values.tolist() if type(i) is str])))
        # tmp += len(list(chain(*[
        #     [k for k in i.split('||') if len(k.split(' ')) > 1]
        #     for i in df_stats['label_intractable'].values.tolist() if type(i) is str])))
        # tmp += len(list(chain(*[
        #     [k for k in i.split('||') if len(k.split(' ')) > 1]
        #     for i in df_stats['label_out_candidates'].values.tolist() if type(i) is str])))
        total_dict[data] = {
            "domain": domain[data],
            "type": types[data],
            "data size": len(df_stats),
            "avg word number": df_stats["n_word"].mean(),
            "avg vocabulary size": df_stats["n_unique_word"].mean(),
            "avg keyword number": df_stats["n_label"].mean(),
            "avg keyword number (tractable only)": df_stats["n_label_in_candidates"].mean(),
            "avg keyword number (tractable only and multiwords)": tmp/len(df_stats),
            "avg vocabulary diversity": df_stats["n_unique_word"].sum()/df_stats["n_word"].sum()
        }
    df = pd.DataFrame(total_dict).T
    print(df)
    df.to_csv('{}/data_statistics.csv'.format(opt.export))
