""" Script to get statistics for `batch_keyphrases.json` file produced from `get_csv_keyphrase.py`

Supposed you have directory like

    ./sample_data
        |- data_0.csv
        |- data_1.csv

Then, you can run the script as,

```
python quality_check/get_aggregate_result.py -a TopicRank
```

You will get

    ./quality_check
        |-statistics
            |-TopicRank
                |-statistics.json

Here, `statistics.json` consisits of a dictionary where the key is stemmed keyphrase and content is following form.
```
{
  "raw": [
    [
      "U.S economy"
    ]
  ],
  "score": [
    0.04537251538994334
  ],
  "count": [
    1
  ],
  "original_sentence_token_size_int": [
    574
  ],
  "dataset_id": [
    "95667"
  ]
}

```
"""


import argparse
import keyphraser
import os
import json
from itertools import chain
from time import time

DEV_MODE = os.getenv('DEV_MODE', '')  # 'without_stopword'


def get_options():
    parser = argparse.ArgumentParser(
            description='',
            formatter_class=argparse.RawTextHelpFormatter
        )
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-a', '--algorithm', help='Name of algorithm, from %s' % keyphraser.list_of_algorithm,
                        required=True, type=str, **share_param)
    parser.add_argument('--output_dir', help='output directory where the results are stored',
                        default='./quality_check/statistics', type=str, **share_param)
    parser.add_argument('--search', help='Searchable mode', action='store_true')
    return parser.parse_args()


def check_stopwords(raw_list: list, search_mode: bool, removed: str):

    stopworder = keyphraser.processing.StopwordDetection()
    if search_mode:
        print('search mode')
        # raw_list = [list(chain(*v["raw"])) for v in json_result.values()]
        # raw_list = list(set(list(chain(*raw_list))))
        print(' - you have %i keywords' % len(raw_list))
        while True:
            inp = input('>>> ')
            if inp == '':
                continue
            elif inp == 'q':
                exit()
            else:
                out = [r for r in raw_list if inp in r]
                for o in out:
                    print(o)
    else:
        print('batch mode')
        if os.path.exists(removed):
            inp = input('There exists file (%s). Overwrite `y` otherwise skip >>>' % removed)
            if inp != 'y':
                return
        out = ''
        start = time()
        print()
        for n, t in enumerate(raw_list):
            print(' - processing: %i/%i (%0.3f sec) \r' % (n, len(raw_list), time() - start), end='', flush=True)
            if stopworder.apply(t) + stopworder.apply_phrase(t) > 0:
                out += ','.join([str(stopworder.apply(t)), str(stopworder.apply_phrase(t)), t])
                out += '\n'
        print()
        with open(removed, 'w') as f:
            f.write(out)


if __name__ == '__main__':
    args = get_options()

    if DEV_MODE == 'without_stopword':
        output_statistics = os.path.join(args.output_dir, args.algorithm, 'statistics_no_stopword.json')
        output_raw = os.path.join(args.output_dir, args.algorithm, 'raw_keyphrases_no_stopword.txt')
        output_removed = os.path.join(args.output_dir, args.algorithm, 'removed_no_stopword.txt')
    else:
        output_statistics = os.path.join(args.output_dir, args.algorithm, 'statistics.json')
        output_raw = os.path.join(args.output_dir, args.algorithm, 'raw_keyphrases.txt')
        output_removed = os.path.join(args.output_dir, args.algorithm, 'removed.txt')

    # with open(output_statistics, 'r') as _f:
    #     output_json = json.load(_f)

    with open(output_raw, 'r') as _f:
        _raw_list = _f.read().split('\n')

    check_stopwords(_raw_list, args.search, output_removed)

