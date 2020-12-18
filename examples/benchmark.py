""" Benchmark preset methods in grapher """
import argparse
import os
import logging
import json
from itertools import permutations
from time import time
from glob import glob
from tqdm import tqdm

import pandas as pd
import grapher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging


def cross_analysis(_export_dir: str):
    all_data = list(map(lambda x: os.path.basename(x) if os.path.isdir(x) else None,
                        glob(os.path.join(_export_dir, '*'))))
    all_data = sorted(list(filter(None, all_data)))
    all_algorithm = ['FirstN', 'LexSpec', 'TFIDF', 'TextRank', 'SingleRank', 'PositionRank', 'LexRank', 'ExpandRank',
                     'SingleTPR', 'TopicRank']
    all_df = []
    for d in all_data:
        tmp_label_dict = {}
        df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
        for a in all_algorithm:
            df[a][a] = 1

            pred_file = os.path.join(_export_dir, d, 'prediction.{}.csv'.format(a))
            pred_df = pd.read_csv(pred_file, index_col=0)
            label = list(map(lambda x: x.split('||'), pred_df['label'].values.tolist()))
            label_pred = list(map(lambda x: str(x).split('||'), pred_df['label_predict'].values.tolist()))
            label_correct = list(map(lambda x: list(set(x[0]).intersection(set(x[1]))), zip(label, label_pred)))
            tmp_label_dict[a] = label_correct

        for a, b in permutations(all_algorithm, 2):
            a_label = tmp_label_dict[a]
            b_label = tmp_label_dict[b]
            label_intersection = list(map(
                lambda x: len(list(set(x[0]).intersection(set(x[1]))))/len(x[0]) if len(x[0]) != 0 else 0,
                zip(a_label, b_label)))
            df[a][b] = sum(label_intersection) / len(label_intersection)
        df.to_csv(os.path.join(_export_dir, d, 'agreement.csv'))
        logging.info('dataset:{} \n {}'.format(d, df))
        all_df.append(df)

    df = pd.DataFrame(columns=all_algorithm, index=all_algorithm)
    for a, b in permutations(all_algorithm, 2):
        df[a][b] = sum(float(i[a][b]) for i in all_df)/len(all_df)
        df[a][a] = 1

    df.to_csv(os.path.join(_export_dir, 'agreement_all.csv'))
    logging.info('All dataset:\n {}'.format(df))


def view_result(_export_dir: str):
    d = 2
    all_data = list(map(lambda x: os.path.basename(x) if os.path.isdir(x) else None,
                        glob(os.path.join(_export_dir, '*'))))
    all_data = sorted(list(filter(None, all_data)))
    all_algorithm = ['FirstN', 'LexSpec', 'TFIDF', 'TextRank', 'SingleRank', 'PositionRank', 'LexRank', 'ExpandRank',
                     'SingleTPR', 'TopicRank'] + ['Best Method']

    df = {i: pd.DataFrame(index=all_data, columns=all_algorithm) for i in ['5', '10', '15', 'time']}
    for i in glob(os.path.join(_export_dir, '*')):
        if not os.path.isdir(i):
            continue
        _data_name = os.path.basename(i)
        _tmp_score = {_n: {} for _n in ['5', '10', '15', 'time']}
        for t in glob(os.path.join(i, 'accuracy.*.json')):
            _algorithm_name = t.split('accuracy.')[-1].replace('.json', '')
            tmp = json.load(open(t))
            for _n in ['5', '10', '15']:
                _score = round(tmp['top_{}'.format(_n)]['f_1'] * 100, d)
                df[_n][_algorithm_name][_data_name] = _score
                _tmp_score[_n][_algorithm_name] = _score
                # df[_n][_algorithm_name][_data_name] = "{0} ({1}/{2})".format(
                #     round(tmp['top_{}'.format(_n)]['f_1'] * 100, d),
                #     round(tmp['top_{}'.format(_n)]['mean_precision'] * 100, d),
                #     round(tmp['top_{}'.format(_n)]['mean_recall'] * 100, d))

            complexity = str(round(tmp['process_time_second'], d))
            df['time'][_algorithm_name][_data_name] = complexity
            _tmp_score['time'][_algorithm_name] = complexity

        for _n in ['5', '10', '15']:
            df[_n]['Best Method'][_data_name] = sorted(_tmp_score[_n].items(), key=lambda x: x[1])[-1][0]
        df['time']['Best Method'][_data_name] = sorted(_tmp_score['time'].items(), key=lambda x: x[1])[0][0]

    return df


def get_options():
    parser = argparse.ArgumentParser(description='Benchmark preset methods in grapher')
    parser.add_argument('-m', '--model', help='model:{}'.format(grapher.VALID_ALGORITHMS), default=None, type=str)
    parser.add_argument('-d', '--data', help='data:{}'.format(grapher.VALID_DATASET_LIST), default=None, type=str)
    parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
    parser.add_argument('-o', '--overwrite', help='overwrite result', action="store_true")
    parser.add_argument('--view', help='view results', action="store_true")
    parser.add_argument('--cross', help='cross algorithm results', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    if opt.view:
        _df = view_result(opt.export)
        for k, v in _df.items():
            logging.info("** Result: {} **\n {}".format(k, v))
            v.to_csv("{}/full-result.{}.csv".format(opt.export, k))
        exit()
    if opt.cross:
        cross_analysis(opt.export)
        exit()

    data_list = grapher.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
    algorithm_list = grapher.VALID_ALGORITHMS if opt.model is None else opt.model.split(',')
    logging.info('Benchmark:\n - dataset: {}\n - algorithm: {}'.format(data_list, algorithm_list))
    for data_name in data_list:
        logging.info('loading data: {}'.format(data_name))
        # load dataset
        data, language = grapher.get_benchmark_dataset(data_name)

        # load model
        for algorithm_name in algorithm_list:

            export_dir = os.path.join(opt.export, data_name)
            accuracy_file = os.path.join(export_dir, 'accuracy.{}.json'.format(algorithm_name))
            prediction_file = os.path.join(export_dir, 'prediction.{}.csv'.format(algorithm_name))
            if os.path.exists(accuracy_file) and os.path.exists(prediction_file) and not opt.overwrite:
                logging.info('skip: found accuracy/prediction file, {}/{}'.format(accuracy_file, prediction_file))
                continue
            
            df_prediction = pd.DataFrame(index=[
                'filename', 'label', 'label_predict', 'score',
                'precision_5', 'precision_10', 'precision_15'])
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
            for n, v in enumerate(tqdm(data)):
                _id = v['id']
                source = v['source']
                gold_keys = v['keywords']   # already stemmed
                # inference
                keys = model.get_keywords(source, n_keywords=15)
                keys_stemmed = [k['stemmed'] for k in keys]

                pred_placeholder = [_id, '||'.join(gold_keys), '||'.join(keys_stemmed),
                                    '||'.join([str(round(k['score'], 3)) for k in keys])]
                for i in [5, 10, 15]:
                    positive_answers = list(set(keys_stemmed[:i]).intersection(set(gold_keys)))
                    tp[str(i)] += len(positive_answers)
                    fn[str(i)] += len(gold_keys) - len(positive_answers)
                    fp[str(i)] += i - len(positive_answers)
                    pred_placeholder.append(len(positive_answers)/i)
                df_prediction[n] = pred_placeholder
            elapsed = time() - start

            # result summary: micro F1 score
            result_json = {'process_time_second': elapsed}
            for i in [5, 10, 15]:
                precision = tp[str(i)]/(fp[str(i)] + tp[str(i)])
                recall = tp[str(i)]/(fn[str(i)] + tp[str(i)])
                f1 = 2 * precision * recall / (precision + recall)
                result_json['top_{}'.format(i)] = {"mean_recall": recall, "mean_precision": precision, "f_1": f1}

            # export
            os.makedirs(export_dir, exist_ok=True)
            with open(accuracy_file, 'w') as f:
                json.dump(result_json, f)
            logging.info(json.dumps(result_json, indent=4, sort_keys=True))

            # export prediction as a csv
            df_prediction.T.to_csv(prediction_file)

            logging.info(' - result exported to {}'.format(export_dir))
