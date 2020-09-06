""" Quality check

check the result for each rows in a csv file sequentially
"""

import keyphraser
import argparse
import pandas as pd


TARGET_STAT_TO_SHOW = ['raw', 'stemmed', 'score', 'count']


def get_options():
    parser = argparse.ArgumentParser(
            description='Check result of csv, one by one.',
            formatter_class=argparse.RawTextHelpFormatter
        )
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-a', '--algorithm', help='Name of algorithm, from %s' % keyphraser.list_of_algorithm,
                        required=True, type=str, **share_param)
    parser.add_argument('-l', '--lan', help='Language `en` or `jp`',
                        default='en', type=str, **share_param)
    parser.add_argument('-n', '--num_keyphrase', help='Number of keyphrases',
                        default=5, type=int, **share_param)
    parser.add_argument('-p', '--path', help='path to csv',
                        default='./sample_data/very_small_subset.csv', type=str, **share_param)
    parser.add_argument('--column_text', help='column of text',
                        default='Story Body', type=str, **share_param)
    parser.add_argument('--column_id', help='column of text',
                        default='dataset_id', type=str, **share_param)
    parser.add_argument('--show_cleaned_str', help='to see the cleaned result.', action='store_true')
    parser.add_argument('--shuffle', help='to shuffle the data.', action='store_true')
    return parser.parse_args()


def load_csv(path_to_file, column_text, column_id):
    df = pd.read_csv(path_to_file, error_bad_lines=False)
    _text = df[column_text].values.tolist()
    _ids = df[column_id].values.astype(str).tolist()
    return _text, _ids


def show_output(__output, additional=None):
    if len(__output) == 0:
        print(__output)
        return
    if additional:
        target = TARGET_STAT_TO_SHOW + additional
    else:
        target = TARGET_STAT_TO_SHOW

    print()
    for o in __output:
        print([(t, o[t]) for t in target])
    print('\nall features:', __output[0].keys())
    print()
    return


if __name__ == '__main__':
    args = get_options()

    cleaner = keyphraser.processing.CleaningStr(language=args.lan)

    extractor = keyphraser.algorithm_instance(args.algorithm, parameters=dict(language=args.lan))
    text, ids = load_csv(args.path, args.column_text, args.column_id)

    index_data = [i for i in range(len(text))]
    if args.shuffle:
        from random import shuffle
        shuffle(index_data)

    n = 0
    while True:
        print('* document: %i/%i' % (n, len(text)))
        single_text = text[index_data[n]]
        single_id = ids[index_data[n]]
        _out = extractor.extract([single_text])

        print('\n----- Original doc -----')
        print(single_text)

        if args.show_cleaned_str:
            print('\n----- Cleaning -----')
            cleaned = cleaner.process(single_text)
            print(cleaned)

        print('\n----- Keyphrases -----')
        show_output(_out[0])

        n += 1
        __inp = input('>>>')
        if __inp == 'q':
            exit()
