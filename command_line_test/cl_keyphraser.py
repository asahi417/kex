""" Keyphrase extraction command line demo """

import graph_keyword_extractor
import argparse
import os


MULTI_METHOD = os.getenv('MULTI_METHOD', 'log')
TARGET_STAT_TO_SHOW = ['raw', 'stemmed', 'score', 'count']


def get_options():
    parser = argparse.ArgumentParser(
            description='This script is to run command line demo for keyphrases extraction.',
            formatter_class=argparse.RawTextHelpFormatter
        )
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-a', '--algorithm', help='Name of algorithm, from %s' % graph_keyword_extractor.list_of_algorithm,
                        required=True, type=str, **share_param)
    parser.add_argument('-l', '--lan', help='Language `en` or `jp`',
                        default='en', type=str, **share_param)
    parser.add_argument('-n', '--num_keyphrase', help='Number of keyphrases',
                        default=5, type=int, **share_param)
    return parser.parse_args()


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

    extractor = graph_keyword_extractor.algorithm_instance(args.algorithm, parameters=dict(language=args.lan))
    if args.multi:
        extractor = graph_keyword_extractor.MultiDocs(extractor, method=MULTI_METHOD)

    n = 0
    docs = []
    while True:
        doc = input('doc %i >>>' % n)

        if doc == 'q':
            exit()

        if doc == '':
            if len(docs) != 0:
                output = extractor.extract(docs, count=args.num_keyphrase)
                show_output(output, additional=['document_frequency'])
            n = 0
            docs = []
        else:
            output = extractor.extract([doc], count=args.num_keyphrase)
            show_output(output[0])
            n = 0
