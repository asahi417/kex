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

Here, `statistics.json` consists of a dictionary where the key is stemmed keyphrase and content is following form.
```
{
  "raw": [
    [
      "World Bank"
    ],
    [
      "World Bank",
      "World Bank"
    ],
    [
      "World Bank"
    ],
    [
      "World Bank",
      "World Bank"
    ],
    [
      "World Bank",
      "World Bank",
      "World Bank"
    ],
    [
      "World Bank",
      "World Bank"
    ],
    [
      "World Bank",
      "World Bank"
    ]
  ],
  "score": [
    0.03150682569650201,
    0.028855972303837917,
    0.047834942855810095,
    0.03484286283856075,
    0.08121496895567953,
    0.062096596907036235,
    0.062096596907036235
  ],
  "count": [
    1,
    2,
    1,
    2,
    3,
    2,
    2
  ],
  "original_sentence_token_size": [
    942,
    913,
    2072,
    804,
    516,
    645,
    645
  ],
  "dataset_id": [
    "65652",
    "65634",
    "69393",
    "92403",
    "94078",
    "94350",
    "94380"
  ]
}
```
"""


import argparse
import keyphraser
import os
import json
import traceback
from time import time
from itertools import chain

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
    return parser.parse_args()


def aggregation(json_result: dict, output_path: str):
    print('* get `statistics.json` file')
    start = time()
    result_dict = dict()
    full_size = len(json_result.items())
    for n, (k, v) in enumerate(json_result.items()):
        try:
            print(' - processing: %i/%i (%0.3f sec) \r' % (n, full_size, time() - start), end='', flush=True)
            for single_v in v:
                stemmed_str = single_v['stemmed']
                raw_list = single_v['raw']
                score_float = single_v['score']
                count_int = single_v['count']
                original_sentence_token_size_int = single_v['original_sentence_token_size']

                if stemmed_str not in result_dict.keys():
                    result_dict[stemmed_str] = dict(
                        raw=[raw_list],
                        score=[score_float],
                        count=[count_int],
                        original_sentence_token_size=[original_sentence_token_size_int],
                        dataset_id=[k]
                    )
                else:
                    result_dict[stemmed_str]['raw'] += [raw_list]
                    result_dict[stemmed_str]['score'] += [score_float]
                    result_dict[stemmed_str]['count'] += [count_int]
                    result_dict[stemmed_str]['original_sentence_token_size'] += [original_sentence_token_size_int]
                    result_dict[stemmed_str]['dataset_id'] += [k]
        except KeyboardInterrupt:
            print()
            print("KeyboardInterrupt")
            break

        except Exception:
            msg = traceback.format_exc()
            print("Error\n - key: %s\n - items: %s" % (k, v))
            print(msg)
    print()
    print(' - saving result.....')

    with open(output_path, 'w') as f:
        json.dump(obj=result_dict, fp=f)

    print('saved at %s' % output_path)


def output_raw_list(json_result: dict, output_path: str):
    raw_list = [list(chain(*v["raw"])) for v in json_result.values()]
    raw_list = list(set(list(chain(*raw_list))))
    raw_list_string = '\n'.join(raw_list)
    with open(output_path, 'w') as f:
        f.write(raw_list_string)


if __name__ == '__main__':
    args = get_options()

    if DEV_MODE == 'without_stopword':
        output_statistics = os.path.join(args.output_dir, args.algorithm, 'statistics_no_stopword.json')
        output_raw = os.path.join(args.output_dir, args.algorithm, 'raw_keyphrases_no_stopword.txt')
        output_dir = os.path.join(args.output_dir, args.algorithm, 'batch_keyphrases_no_stopword.json')
    else:
        output_statistics = os.path.join(args.output_dir, args.algorithm, 'statistics.json')
        output_raw = os.path.join(args.output_dir, args.algorithm, 'raw_keyphrases.txt')
        output_dir = os.path.join(args.output_dir, args.algorithm, 'batch_keyphrases.json')

    # output statistics
    if os.path.exists(output_statistics):
        inp = input('There exists file (%s). Overwrite `y` otherwise skip >>>' % output_statistics)
        if inp == 'y':
            flg = True
        else:
            flg = False
    else:
        flg = True
    if flg:
        with open(output_dir, 'r') as _f:
            results = json.load(_f)
        aggregation(results, output_statistics)

    # output raw token list
    if os.path.exists(output_raw):
        inp = input('There exists file (%s). Overwrite `y` otherwise skip >>>' % output_raw)
        if inp == 'y':
            flg = True
        else:
            flg = False
    else:
        flg = True
    if flg:
        with open(output_statistics, 'r') as _f:
            stat = json.load(_f)
        output_raw_list(stat, output_raw)
