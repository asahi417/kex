""" Script to get keyphrase for csv file under specified directory

Supposed you have directory like

    ./sample_data
        |- data_0.csv
        |- data_1.csv

Then, you can run the script as,

```
python quality_check/get_csv_keyphrase.py -a TopicRank --dir ./sample_data
```

You will get

    ./quality_check
        |-statistics
            |-TopicRank
                |-batch_keyphrases.json
                |-batch_errors.json

Here, `batch_keyphrases.json` consisits of a dictionary where the key is dataset_id and content is following form.
```
[
  {
    "raw": [
      "series"
    ],
    "stemmed": "seri",
    "score": 0.11177754643075341,
    "count": 1,
    "original_sentence_token_size": 54
  },
  {
    "raw": [
      "stock"
    ],
    "stemmed": "stock",
    "score": 0.10543739709725984,
    "count": 1,
    "original_sentence_token_size": 54
  },
  {
    "raw": [
      "Source text"
    ],
    "stemmed": "sourc text",
    "score": 0.10260031326821299,
    "count": 1,
    "original_sentence_token_size": 54
  },
  {
    "raw": [
      "Cynvenio Biosystems"
    ],
    "stemmed": "cynvenio biosystem",
    "score": 0.10167003244102098,
    "count": 1,
    "original_sentence_token_size": 54
  },
  {
    "raw": [
      "Chinese"
    ],
    "stemmed": "chines",
    "score": 0.09971209604082434,
    "count": 1,
    "original_sentence_token_size": 54
  }
]
```
"""


import argparse
import keyphraser
import pandas as pd
import os
import json
import traceback
import random
from glob import glob
from time import time

DEV_MODE = os.getenv('DEV_MODE', '')  # 'without_stopword'
TARGET_STAT_TO_SHOW = ['raw', 'stemmed', 'score', 'count', 'original_sentence_token_size']


def get_options():
    parser = argparse.ArgumentParser(
            description='',
            formatter_class=argparse.RawTextHelpFormatter
        )
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-a', '--algorithm', help='Name of algorithm, from %s' % keyphraser.list_of_algorithm,
                        required=True, type=str, **share_param)
    parser.add_argument('-l', '--lan', help='Language `en` or `jp`',
                        default='en', type=str, **share_param)
    parser.add_argument('-n', '--num_keyphrase', help='Number of keyphrases',
                        default=5, type=int, **share_param)
    parser.add_argument('-b', '--batch_size', help='Batch size',
                        default=50, type=int, **share_param)
    parser.add_argument('--seed', help='Random seed',
                        default=4, type=int, **share_param)
    parser.add_argument('--dir', help='directory where the csv files are',
                        default='./sample_data/', type=str, **share_param)
    parser.add_argument('--output_dir', help='output directory where the results are stored',
                        default='./quality_check/statistics', type=str, **share_param)
    parser.add_argument('--file_size',
                        help='file size to get result (if csv files are found more than this number,'
                             'will randomly sample csv file)',
                        default=10, type=int, **share_param)
    parser.add_argument('--column_text', help='column of text',
                        default='Story Body', type=str, **share_param)
    parser.add_argument('--column_id', help='column of text',
                        default='dataset_id', type=str, **share_param)
    # parser.add_argument('--refresh', help='refresh statistics (re-calculation)', action='store_true')
    return parser.parse_args()


def load_csv(path_to_file, column_text, column_id):
    df = pd.read_csv(path_to_file, error_bad_lines=False)
    _text = df[column_text].values.tolist()
    _id = df[column_id].values.astype(str).tolist()
    return _text, _id


def analyse_single_file(texts,
                        ids,
                        output_dir,
                        algorithm_instance,
                        num_keyphrase: int,
                        batch_size: int=10,
                        refresh: bool=True):

    def get_kp(batch_doc: list, batch_id: list):
        phrases_result = algorithm_instance.extract(batch_doc, count=num_keyphrase)
        assert len(batch_id) == len(phrases_result)

        def __format(__dict):
            return dict([(i, __dict[i]) for i in TARGET_STAT_TO_SHOW])

        __output = dict([
            (__id, [__format(__p) for __p in phrase] if len(phrase) != 0 else [])
            for __id, phrase in zip(batch_id, phrases_result)])
        return __output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if DEV_MODE == 'without_stopword':
        output_keyphrase = os.path.join(output_dir, 'batch_keyphrases_no_stopword.json')
        output_errors = os.path.join(output_dir, 'batch_errors_no_stopword.json')
    else:
        output_keyphrase = os.path.join(output_dir, 'batch_keyphrases.json')
        output_errors = os.path.join(output_dir, 'batch_errors.json')

    errors = []
    if refresh:
        output_json = dict()
    else:
        if os.path.exists(output_keyphrase):
            output_json = json.load(open(output_keyphrase, 'r'))
        else:
            output_json = dict()

    __batch_doc = list()
    __batch_id = list()
    start = time()
    for n, (_text, _id) in enumerate(zip(texts, ids)):

        if _id in output_json.keys():
            continue

        __batch_doc.append(_text)
        __batch_id.append(_id)
        try:
            if len(__batch_doc) >= batch_size:
                print(' - processing: %i/%i (%0.3f sec) \r' % (n, len(texts), time() - start), end='', flush=True)
                _tmp_output = get_kp(__batch_doc, __batch_id)
                output_json.update(_tmp_output)
                __batch_doc = list()
                __batch_id = list()
        except KeyboardInterrupt:
            print()
            print("KeyboardInterrupt")
            break

        except Exception:
            msg = traceback.format_exc()
            print()
            print('  >>> Error at %i: <<<' % n)
            print(msg)
            errors.append(dict(msg=msg, data_id=__batch_id, doc=__batch_doc))
            __batch_doc = list()
            __batch_id = list()

    print()
    print(' - saving result.....')

    with open(output_errors, 'w') as f:
        json.dump(obj=dict(errors=errors), fp=f)

    with open(output_keyphrase, 'w') as f:
        json.dump(obj=output_json, fp=f)

    print('saved at %s, %s' % (output_keyphrase, output_errors))
    print('there were %i errors.' % len(errors))


if __name__ == '__main__':
    args = get_options()

    extractor = keyphraser.algorithm_instance(args.algorithm, parameters=dict(language=args.lan))

    files = glob(os.path.join(args.dir, '*.csv'))
    random.Random(args.seed).shuffle(files)
    files = files[:min(len(files), args.file_size)]
    print('Total %i files found' % len(files))
    for _n, __file in enumerate(files):
        print('* file %i: (%s)' % (_n, __file))
        _texts, _ids = load_csv(__file, args.column_text, args.column_id)
        _output_dir = os.path.join(args.output_dir, args.algorithm)

        # refresh when first file if args.refresh is True
        # _refresh = args.refresh if _n == 0 else False
        analyse_single_file(_texts,
                            _ids,
                            output_dir=_output_dir,
                            algorithm_instance=extractor,
                            num_keyphrase=args.num_keyphrase,
                            batch_size=args.batch_size,
                            refresh=False)
