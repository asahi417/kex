""" Benchmark preset methods in grapher """
import argparse
import os
import logging
import json
from typing import List
from itertools import permutations
from glob import glob
from tqdm import tqdm
from itertools import product

import pandas as pd
import grapher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
n_keywords = 100000


def get_data_algorithm(_export_dir):
    all_data = sorted(list(filter(
        None,
        map(lambda x: os.path.basename(x) if os.path.isdir(x) else None,
            glob(os.path.join(_export_dir, '*')))
    )))

    all_algorithm = list(map(
        lambda x: x.split('.')[-2],
        glob(os.path.join(_export_dir, '*/accuracy.*.json'))
    ))
    all_algorithm = list(filter(lambda x: x in all_algorithm, grapher.VALID_ALGORITHMS))
    return all_data, all_algorithm


def aggregate_agreement(_export_dir: str, top_n: int = 5):

    def clip(_list):
        return _list[:min(len(_list), top_n)]

    all_data, all_algorithm = get_data_algorithm(_export_dir)
    all_df = []
    for d in all_data:
        tmp_label_dict = {}
        df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
        for a in all_algorithm:
            df[a][a] = 100.0
            pred_df = pd.read_csv(os.path.join(_export_dir, d, 'prediction.{}.csv'.format(a)), index_col=0)
            tmp_label_dict[a] = list(map(lambda x: clip(str(x).split('||')), pred_df['label_predict'].values.tolist()))

        for a, b in permutations(all_algorithm, 2):
            label_intersection = list(map(
                lambda x: len(list(set(x[0]).intersection(set(x[1])))) / len(x[0]) if len(x[0]) != 0 else 0,
                zip(tmp_label_dict[a], tmp_label_dict[b])))
            df[a][b] = round(sum(label_intersection) / len(label_intersection) * 100, 1)
        df.to_csv(os.path.join(_export_dir, d, 'agreement.csv'))
        logging.info('dataset:{} \n {}'.format(d, df))
        all_df.append(df)

    df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
    for a, b in permutations(all_algorithm, 2):
        df[a][b] = round(sum(float(i[a][b]) for i in all_df)/len(all_df), 1)
        df[a][a] = 100.0

    df.to_csv(os.path.join(_export_dir, 'agreement_all.csv'))
    logging.info('All dataset:\n {}'.format(df))


def aggregate_result(_export_dir: str, d: int = 1):
    all_data, all_algorithm = get_data_algorithm(_export_dir)

    with open(glob(os.path.join(_export_dir, '*/accuracy.*.json'))[0], 'r') as f:
        tmp = json.load(f)
        metrics = list(tmp.keys())
        metrics.pop(metrics.index('mrr'))

    def update(_df, _key):
        if _key not in _df.keys():
            _df[_key] = pd.DataFrame(index=all_data, columns=all_algorithm)
        return _df

    df = {}
    df_full = {}
    for i in glob(os.path.join(_export_dir, '*/accuracy.*.json')):
        _data_name = i.split('/')[-2]
        _algorithm_name = i.split('accuracy.')[-1].replace('.json', '')
        tmp = json.load(open(i))
        df = update(df, 'mrr')
        df['mrr'][_algorithm_name][_data_name] = round(tmp['mrr'] * 100, d)
        for _n in metrics:
            _f1 = round(tmp[_n]['f_1'] * 100, d)
            _pre = round(tmp[_n]['mean_precision'] * 100, d)
            _rec = round(tmp[_n]['mean_recall'] * 100, d)
            df = update(df, '{}.f1'.format(_n))
            df = update(df, '{}.precision'.format(_n))
            df = update(df, '{}.recall'.format(_n))
            df['{}.f1'.format(_n)][_algorithm_name][_data_name] = _f1
            df['{}.precision'.format(_n)][_algorithm_name][_data_name] = _pre
            df['{}.recall'.format(_n)][_algorithm_name][_data_name] = _rec
            df_full = update(df_full, _n)
            df_full[_n][_algorithm_name][_data_name] = '{} ({}, {})'.format(_f1, _pre, _rec)
    for _k, _v in df.items():
        logging.info("** Result: {} **\n {}".format(_k, _v))
        _v.to_csv("{}/result.{}.csv".format(_export_dir, _k))
    for _k, _v in df_full.items():
        logging.info("** Result: {} **\n {}".format(_k, _v))
        _v.to_csv("{}/result.{}.combined.csv".format(_export_dir, _k))


def get_model_prediction(model_name: str, data_name: str):
    """ get prediction from single model on single dataset """
    data, language = grapher.get_benchmark_dataset(data_name)
    model = grapher.AutoAlgorithm(model_name, language=language)

    # compute prior
    if model.prior_required:
        logging.info(' - computing prior...')
        try:
            model.load('./cache/{}/priors'.format(data_name))
        except Exception:
            model.train([i['source'] for i in data], export_directory='./cache/{}/priors'.format(data_name))
    preds, labels, scores, ids = [], [], [], []
    for n, v in enumerate(tqdm(data)):
        ids.append(v['id'])
        # inference
        keys = model.get_keywords(v['source'], n_keywords=n_keywords)
        preds.append([k['stemmed'] for k in keys])
        scores.append([k['score'] for k in keys])
        labels.append(v['keywords'])  # already stemmed
    return preds, labels, scores, ids


def run_benchmark(data: (List, str) = None,
                  model: (List, str) = None,
                  export_dir_root: str = './benchmark_result',
                  top_n: List = None,
                  fixed_metric: bool = True):
    """ Run keyword extraction benchmark """
    _prefix = '' if fixed_metric else '.unfixed.'
    if top_n is None:
        top_n = [5, 10, 15]
    if data is None:
        data_list = grapher.VALID_DATASET_LIST
    else:
        data_list = data if type(data) is list else [data]
    if model is None:
        model_list = grapher.VALID_ALGORITHMS
    else:
        model_list = model if type(model) is list else [model]

    logging.info('Benchmark:\n - dataset: {}\n - algorithm: {}'.format(data_list, model_list))
    for data_name, model_name in product(data_list, model_list):
        logging.info(' - algorithm: {}\n - data: {}'.format(model_name, data_name))
        export_dir = os.path.join(export_dir_root, data_name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        accuracy_file = os.path.join(export_dir, 'accuracy.{}{}.json'.format(model_name, _prefix))
        prediction_file = os.path.join(export_dir, 'prediction.{}{}.csv'.format(model_name, _prefix))
        if os.path.exists(accuracy_file) and os.path.exists(prediction_file):
            logging.info('skip: found accuracy/prediction file, `{}`, `{}`'.format(accuracy_file, prediction_file))
            continue
        if os.path.exists(prediction_file):
            df_prediction_cached = pd.read_csv(prediction_file)
            labels = [i.split('||') if type(i) is str else '' for i in df_prediction_cached['label'].values.tolist()]
            preds = [i.split('||') if type(i) is str else '' for i in df_prediction_cached['label_predict'].values.tolist()]
        else:
            preds, labels, scores, ids = get_model_prediction(model_name, data_name)
            df_prediction = pd.DataFrame([
                [i, '||'.join(l), '||'.join(p), '||'.join(list(map(lambda x: str(round(x, 3)), s)))]
                for p, l, s, i in zip(preds, labels, scores, ids)],
                columns=['filename', 'label', 'label_predict', 'score'])
            df_prediction.to_csv(prediction_file)

        # run algorithm and test it over data
        tp = {n: 0 for n in list(map(str, top_n))}
        fn = {n: 0 for n in list(map(str, top_n))}
        fp = {n: 0 for n in list(map(str, top_n))}
        mrr = []
        for n_, (l_, p) in enumerate(zip(labels, preds)):
            all_ranks = [n + 1 for n, p_ in enumerate(p) if p_ in l_]
            if len(all_ranks) == 0:  # no answer is found (TopicRank may cause it)
                mrr.append(1 / (len(preds) + 1))
            else:
                mrr.append(1/min(all_ranks))
            for i in tp.keys():
                if fixed_metric:
                    n = min(int(i), len(l_))
                else:
                    n = int(i)
                positive_answers = list(set(p[:n]).intersection(set(l_)))
                tp[i] += len(positive_answers)
                fn[i] += len(l_) - len(positive_answers)
                fp[i] += n - len(positive_answers)

        # result summary: micro F1 score
        result_json = {'mrr': sum(mrr)/len(mrr)}
        for i in tp.keys():
            precision = tp[str(i)] / (fp[str(i)] + tp[str(i)])
            recall = tp[str(i)] / (fn[str(i)] + tp[str(i)])
            f1 = 2 * precision * recall / (precision + recall)
            result_json[i] = {"mean_recall": recall, "mean_precision": precision, "f_1": f1}

        # export
        os.makedirs(export_dir, exist_ok=True)
        with open(accuracy_file, 'w') as f:
            json.dump(result_json, f)
        logging.info(json.dumps(result_json, indent=4, sort_keys=True))
        logging.info(' - result exported to {}'.format(export_dir))


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in grapher')
    parser.add_argument('-m', '--model', help='model:{}'.format(grapher.VALID_ALGORITHMS), default=None, type=str)
    parser.add_argument('-d', '--data', help='data:{}'.format(grapher.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    parser.add_argument('--top-n', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    run_benchmark(data=opt.data.split(',') if opt.data is not None else opt.data,
                  model=opt.model.split(',') if opt.model is not None else opt.model,
                  export_dir_root=opt.export,
                  top_n=opt.top_n.split(',') if opt.top_n is not None else opt.top_n,
                  fixed_metric=False)
    # aggregate_result(opt.export)
    # aggregate_agreement(opt.export)
