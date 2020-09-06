""" Pre-processing command line demo """

import keyphraser
import argparse


def get_options():
    parser = argparse.ArgumentParser(
            description='This script is to run command line demo for keyphrases extraction.',
            formatter_class=argparse.RawTextHelpFormatter
        )
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-l', '--lan', help='Language `en` or `jp`',
                        default='en', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options()
    processor = keyphraser.processing.Processing(language=args.lan)
    cleaner = keyphraser.processing.CleaningStr(language=args.lan)

    while True:
        doc = input('doc >>>')

        if doc == 'q':
            exit()

        elif doc == '':
            continue

        else:
            print()
            print('----- Cleaning -----')
            cleaned = cleaner.process(doc)
            print(cleaned)
            if args.lan == 'en':
                print('----- Spacy -----')
                for n, spacy_object in enumerate(processor.spacy_processor(cleaned)):
                    print('- text: %s' % spacy_object.text)
                    print('- lemm: %s' % spacy_object.lemma_)
                    print('- stem: %s' % processor.stemmer(spacy_object.text))
                    print('- pos : %s' % spacy_object.pos_)
                    print()
            print('----- Processor -----')
            processed = processor(doc)
            for k, v in processed.items():
                print(k, v)
                print()
            print()



