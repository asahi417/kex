""" Stopword detection command line demo """

import graph_keyword_extractor


if __name__ == '__main__':

    stopworder = graph_keyword_extractor.processing.StopwordDetection()
    skip_phraser = graph_keyword_extractor.processing.SkipPhrase()

    print('Interactive mode')

    while True:
        doc = input('doc >>>')

        if doc == 'q':
            exit()

        elif doc == '':
            continue

        else:
            print()
            out = stopworder.apply(doc, debug=True)
            print('stopword:', out)
            print()

            out = stopworder.apply_phrase(doc, debug=True)
            print('stopphrase:', out)
            print()

            out = skip_phraser.apply(doc, debug=True)
            print('skip_phraser:', out)
            print()


