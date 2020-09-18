import json
import os
from itertools import chain

import untangle

CACHE_DIR = './cache'
__all__ = 'get_benchmark_dataset'


def decode_xml(_file):
    doc = untangle.parse(open(_file))
    all_sentences = doc.root.document.sentences.sentence
    reconstruct_sentences = []
    for sent in all_sentences:
        reconstruct_sentences += [' '.join([t.word.cdata for t in sent.tokens.token])]
    reconstruct_sentences = ' '.join(reconstruct_sentences)
    return reconstruct_sentences


def get_benchmark_dataset(data: str = 'SemEval2010', cache_dir: str = None):
    """

     Parameter
    -------------
    data: str
        dataset name
    cache_dir: str
        directory to cache the data

     Return
    -------------
    dictionary consists of the data
    flag, True if the ground truth is already stemmed
    """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    if data == 'SemEval2010':
        if not os.path.exists('{}/semeval-2010-pre-master'.format(cache_dir)):
            os.system('wget -O {}/master.zip https://github.com/boudinfl/semeval-2010-pre/archive/master.zip'.
                      format(cache_dir))
            os.system('unzip {0}/master.zip -d {0}'.format(cache_dir))
        gold_keys = json.load(open(
            '{}/semeval-2010-pre-master/references/test.combined.stem.json'.format(cache_dir), 'r'))
        answer_dict = [{
            'keywords': list(chain(*v)),
            'source': decode_xml('{0}/semeval-2010-pre-master/test/lvl-4/{1}.xml'.format(cache_dir, k)),
            'id': k} for k, v in gold_keys.items()]
        flag_stemmed = True
    else:
        raise ValueError('undefined data name: {}'.format(data))
    return answer_dict, flag_stemmed



