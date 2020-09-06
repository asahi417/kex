""" Extract sentence from xml output formed by StanfordParser """

import json
import untangle
from glob import glob
from itertools import chain


ESCAPED_PUNCTUATION = {
    '-lrb-': '(', '-LRB-': '(',
    '-rrb-': ')', '-RRB-': ')',
    '-lsb-': '[', '-LSB-': '[',
    '-rsb-': ']', '-RSB-': ']',
    '-lcb-': '{', '-LCB-': '{',
    '-rcb-': '}', '-RCB-': '}'
}


class BatcherXML:
    """ XML data batcher

     Usage
    ------------------
    >>> batch = BatcherXML(path_reference='test.reader.json',
    ...                    path_reference_stemmed='test.reader.stem.json',
    ...                    path_xml='test')
    >>> for data in batch:
    >>>     raw_docs = data['raw_docs']  # string
    >>>     reference = data['reference']  # list of string (reference)
    >>>     reference_stem = data['reference_stem']  # list of string (stemmed reference)
    """

    def __init__(self,
                 path_reference: str,
                 path_reference_stemmed: str,
                 path_xml: str):
        """ Data batcher for xml data

        :param path_reference:
        :param path_reference_stemmed:
        """

        # reference
        self.__reference = json.load(open(path_reference))
        self.__reference_stemmed = json.load(open(path_reference_stemmed))

        self.__list_xmls = glob("%s/*.xml" % path_xml)

    def __iter__(self):
        self.__data_size = len(self.__list_xmls)
        self.__id = -1
        return self

    def __next__(self):
        self.__id += 1
        if self.__id >= self.__data_size:
            raise StopIteration

        tmp = self.__list_xmls[self.__id]
        record_id = tmp.replace('.xml', '').split('/')[-1]
        ref = list(chain(*self.__reference[record_id]))
        ref_stemmed = list(chain(*self.__reference_stemmed[record_id]))

        obj = untangle.parse(tmp)
        sentences = obj.root.document.sentences.sentence
        list_of_sent = [' '.join([self.escaped_punctuation(token.word.cdata) for token in tokens.tokens.token])
                        for tokens in sentences]
        raw_docs = ' '.join(list_of_sent)
        value = dict(document=raw_docs, reference=ref, reference_stem=ref_stemmed)
        return value

    @staticmethod
    def escaped_punctuation(token):
        if token in ESCAPED_PUNCTUATION.keys():
            return ESCAPED_PUNCTUATION[token]
        else:
            return token
