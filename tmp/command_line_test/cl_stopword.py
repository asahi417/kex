""" Stopword detection command line demo """

import keyphraser


if __name__ == '__main__':

    stopworder = keyphraser.processing.StopwordDetection()
    skip_phraser = keyphraser.processing.SkipPhrase()

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


