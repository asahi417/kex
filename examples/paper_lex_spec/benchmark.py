""" Benchmark algorithms over built-in dataset """
import argparse
import os
import logging
import json
from typing import List
from itertools import permutations
from glob import glob
from tqdm import tqdm
from itertools import product

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import kex

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
top_n = [5, 10]
export_dir_root = './benchmark'
os.makedirs(export_dir_root, exist_ok=True)


def get_data_algorithm():
    all_data = sorted(list(filter(
        None,
        map(lambda x: os.path.basename(x) if os.path.isdir(x) else None,
            glob(os.path.join(export_dir_root, '*')))
    )))

    all_algorithm = list(map(
        lambda x: x.split('.')[-2],
        glob(os.path.join(export_dir_root, '*/accuracy.*.json'))
    ))
    all_algorithm = list(filter(lambda x: x in all_algorithm, kex.VALID_ALGORITHMS))
    return all_data, all_algorithm


def aggregate_agreement(top_n_prediction: int = 5):

    def clip(_list):
        return _list[:min(len(_list), top_n_prediction)]

    all_data, all_algorithm = get_data_algorithm()
    all_df = []
    for d in all_data:
        tmp_label_dict = {}
        df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
        for a in all_algorithm:
            df[a][a] = 100.0
            pred_df = pd.read_csv(os.path.join(export_dir_root, d, 'prediction.{}.csv'.format(a)), index_col=0)
            tmp_label_dict[a] = list(map(lambda x: clip(str(x).split('||')), pred_df['label_predict'].values.tolist()))

        for a, b in permutations(all_algorithm, 2):
            label_intersection = list(map(
                lambda x: len(list(set(x[0]).intersection(set(x[1])))) / len(x[0]) if len(x[0]) != 0 else 0,
                zip(tmp_label_dict[a], tmp_label_dict[b])))
            df[a][b] = sum(label_intersection) / len(label_intersection) * 100
        df.to_csv(os.path.join(export_dir_root, d, 'agreement.csv'))
        logging.info('dataset:{} \n {}'.format(d, df))
        all_df.append(df)

    df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
    for a, b in permutations(all_algorithm, 2):
        df[a][b] = sum(float(i[a][b]) for i in all_df)/len(all_df)
        df[a][a] = 100

    df.to_csv(os.path.join(export_dir_root, 'agreement_all.csv'))
    logging.info('All dataset:\n {}'.format(df))

    # plot heatmap
    fig = plt.figure()
    fig.clear()
    df = df.astype(float).round()
    # sns_plot = sns.heatmap(df, annot=True, fmt="g", cmap='viridis', cbar=True)
    sns_plot = sns.heatmap(df, annot=True, fmt="g", cbar=True)
    sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=60)
    sns_plot.tick_params(labelsize=10)
    fig = sns_plot.get_figure()
    plt.tight_layout()
    fig.savefig('{}/agreement_all.heatmap.png'.format(export_dir_root))
    fig.savefig('{}/agreement_all.heatmap.pdf'.format(export_dir_root))


def aggregate_result(d: int = 1):
    all_data, all_algorithm = get_data_algorithm()
    metrics = list(map(str, top_n))

    def update(_df, _key):
        if _key not in _df.keys():
            _df[_key] = pd.DataFrame(index=all_data, columns=all_algorithm)
        return _df

    df = {}
    df_full = {}
    for i in glob(os.path.join(export_dir_root, '*/accuracy.*.json')):
        _data_name = i.split('/')[-2]
        _algorithm_name = i.split('accuracy.')[-1].replace('.json', '')
        tmp = json.load(open(i))
        df = update(df, 'mrr')
        df['mrr'][_algorithm_name][_data_name] = tmp['mrr'] * 100
        for _n in metrics:
            for _type in ['fixed', 'unfixed']:
                print(i, tmp)
                df = update(df, '{}.f1.{}'.format(_n, _type))
                df = update(df, '{}.precision.{}'.format(_n, _type))
                df = update(df, '{}.recall.{}'.format(_n, _type))
                df['{}.f1.{}'.format(_n, _type)][_algorithm_name][_data_name] = tmp[_type][_n]['f_1'] * 100
                df['{}.precision.{}'.format(_n, _type)][_algorithm_name][_data_name] = tmp[_type][_n]['precision'] * 100
                df['{}.recall.{}'.format(_n, _type)][_algorithm_name][_data_name] = tmp[_type][_n]['recall'] * 100
    for _k, _v in df.items():
        logging.info("** Result: {} **\n {}".format(_k, _v))
        _v.to_csv("{}/result.{}.csv".format(export_dir_root, _k))
    for _k, _v in df_full.items():
        logging.info("** Result: {} **\n {}".format(_k, _v))
        _v.to_csv("{}/result.{}.csv".format(export_dir_root, _k))


def get_model_prediction(model_name: str, data_name: str):
    """ get prediction from single model on single dataset """
    data, language = kex.get_benchmark_dataset(data_name)
    model = kex.get_algorithm(model_name, language=language)

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
        keys = model.get_keywords(v['source'], n_keywords=100000)
        preds.append([k['stemmed'] for k in keys])
        scores.append([k['score'] for k in keys])
        labels.append(v['keywords'])  # already stemmed
    return preds, labels, scores, ids


def run_benchmark():
    """ Run keyword extraction benchmark """
    data_list = kex.VALID_DATASET_LIST
    model_list = kex.VALID_ALGORITHMS
    logging.info('Benchmark:\n - dataset: {}\n - algorithm: {}'.format(data_list, model_list))
    for data_name, model_name in product(data_list, model_list):
        logging.info(' - algorithm: {}\n - data: {}'.format(model_name, data_name))
        export_dir = os.path.join(export_dir_root, data_name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        accuracy_file = os.path.join(export_dir, 'accuracy.{}.json'.format(model_name))
        prediction_file = os.path.join(export_dir, 'prediction.{}.csv'.format(model_name))
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
        tp_unfixed = {n: 0 for n in list(map(str, top_n))}
        fn_unfixed = {n: 0 for n in list(map(str, top_n))}
        fp_unfixed = {n: 0 for n in list(map(str, top_n))}
        mrr = []
        for n_, (l_, p) in enumerate(zip(labels, preds)):
            all_ranks = [n + 1 for n, p_ in enumerate(p) if p_ in l_]
            if len(all_ranks) == 0:  # no answer is found (TopicRank may cause it)
                mrr.append(1 / (len(preds) + 1))
            else:
                mrr.append(1 / min(all_ranks))
            for i in tp.keys():
                n = min(int(i), len(l_))
                n_unfixed = int(i)

                positive_answers = list(set(p[:n]).intersection(set(l_)))
                tp[i] += len(positive_answers)
                fn[i] += len(l_) - len(positive_answers)
                fp[i] += n - len(positive_answers)

                positive_answers = list(set(p[:n_unfixed]).intersection(set(l_)))
                tp_unfixed[i] += len(positive_answers)
                fn_unfixed[i] += len(l_) - len(positive_answers)
                fp_unfixed[i] += n_unfixed - len(positive_answers)

        # result summary: micro F1 score
        result_json = {'mrr': sum(mrr)/len(mrr), 'fixed': {}, 'unfixed': {}}
        for i in tp.keys():

            precision = tp[str(i)] / (fp[str(i)] + tp[str(i)])
            recall = tp[str(i)] / (fn[str(i)] + tp[str(i)])
            f1 = 2 * precision * recall / (precision + recall)
            result_json['fixed'][i] = {"recall": recall, "precision": precision, "f_1": f1}

            precision = tp_unfixed[str(i)] / (fp_unfixed[str(i)] + tp_unfixed[str(i)])
            recall = tp_unfixed[str(i)] / (fn_unfixed[str(i)] + tp_unfixed[str(i)])
            f1 = 2 * precision * recall / (precision + recall)
            result_json['unfixed'][i] = {"recall": recall, "precision": precision, "f_1": f1}

        # export
        os.makedirs(export_dir, exist_ok=True)
        with open(accuracy_file, 'w') as f:
            json.dump(result_json, f)
        logging.info(json.dumps(result_json, indent=4, sort_keys=True))
        logging.info(' - result exported to {}'.format(export_dir))


if __name__ == '__main__':
    run_benchmark()
    aggregate_result()
    aggregate_agreement()
    mrr = pd.read_csv("{}/result.mrr.csv".format(export_dir_root), index_col=0)
    mrr['metric'] = 'mrr'
    pre5 = pd.read_csv("{}/result.5.precision.fixed.csv".format(export_dir_root), index_col=0)
    pre5['metric'] = 'pre5'
    pre10 = pd.read_csv("{}/result.10.precision.fixed.csv".format(export_dir_root), index_col=0)
    pre10['metric'] = 'pre10'
    pd.concat([pre5, mrr, pre10]).to_csv("{}/result.paper.csv".format(export_dir_root))
