import os
import re
import logging
import requests
import zipfile
import shutil
from glob import glob
from ._phrase_constructor import PhraseConstructor
from nltk.stem.porter import PorterStemmer  # only for English

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

CACHE_DIR = '{}/cache_kex'.format(os.path.expanduser('~'))
STEMMER = PorterStemmer()
VALID_DATASET_LIST = [
    # "110-PT-BN-KP", "cacic", "pak2018", "WikiNews", "wicc"
    "citeulike180", "fao30", "fao780", "Inspec", "kdd", "Krapivin2009", "Nguyen2007", "PubMed", "Schutz2008",
    "SemEval2010", "SemEval2017", "theses100", "wiki20", "www", "500N-KPCrowd-v1.1"
]

__all__ = ('get_benchmark_dataset', "VALID_DATASET_LIST")


def preprocess_source(string):
    escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']', '-lcb-': '{', '-rcb-': '}'}
    for k, v in escaped_punctuation.items():
        string = string.replace(k, v)
    return string


def split_for_keywords(string):

    def cleaner(_string):
        _string = re.sub(r'\A\s*', '', _string)
        _string = re.sub(r'\s*\Z', '', _string)
        _string = _string.replace('\n', ' ').replace('\t', '')
        _string = re.sub(r'\s{2,}', ' ', _string)
        _string = ' '.join(list(map(lambda x: STEMMER.stem(x), _string.split(' '))))
        return _string

    keys = list(filter(
        lambda x: len(x) > 0, [cleaner(s) for s in re.split('[\n,]', string)])
    )
    return list(set(keys))


def get_benchmark_dataset(data: str = 'Inspec',
                          cache_dir: str = None,
                          keep_only_valid_label: bool = False):
    """ Get a dataset for keyword extraction (labels are not stemmed)

     Parameter
    -------------
    data: str
        dataset name
    cache_dir: str
        directory to cache the data
    keep_only_valid_label: bool
        False to get all the label, otherwise only keep the label set that is in the phrase candidates

     Return
    -------------
    dictionary consists of the data
    language
    """
    assert data in VALID_DATASET_LIST, "undefined dataset: {}".format(data)
    url = "https://github.com/asahi417/KeywordExtractor-Datasets/raw/master/datasets"
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists("{}/{}".format(cache_dir, data)):
        with open('{}/{}.zip'.format(cache_dir, data), "wb") as f:
            r = requests.get('{}/{}.zip'.format(url, data))
            f.write(r.content)
        if data == "110-PT-BN-KP":
            with zipfile.ZipFile('{}/{}.zip'.format(cache_dir, data), 'r') as zip_ref:
                zip_ref.extractall('{}/{}'.format(cache_dir, data))
        elif data == "WikiNews":
            with zipfile.ZipFile('{}/{}.zip'.format(cache_dir, data), 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            shutil.move('{}/WKC'.format(cache_dir), '{}/{}'.format(cache_dir, data))
        else:
            with zipfile.ZipFile('{}/{}.zip'.format(cache_dir, data), 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
    if not os.path.exists("{}/{}".format(cache_dir, data)):
        raise ValueError('data `{}` is not in the list {}'.format(data, url))

    def safe_open(_file):
        with open(_file, 'r') as f:
            return f.read()

    language = safe_open("{}/{}/language.txt".format(cache_dir, data)).replace('\n', '')
    answer_dict = [{
        "keywords":
            split_for_keywords(safe_open(
                "{}/{}/keys/{}.key".format(cache_dir, data, '.txt'.join(os.path.basename(t).split('.txt')[:-1])))),
        "source": preprocess_source(safe_open(t)),
        "id": os.path.basename(t)} for t in glob("{}/{}/docsutf8/*.txt".format(cache_dir, data))]

    if not keep_only_valid_label:
        return answer_dict, language

    phraser = PhraseConstructor()

    def _filter_label(_dict):
        phrase, stemmed_token = phraser.tokenize_and_stem_and_phrase(_dict['source'])
        keywords_valid = list(set(phrase.keys()).intersection(set(_dict['keywords'])))
        if len(keywords_valid) == 0:
            return None
        else:
            _dict['keywords'] = keywords_valid
            return _dict

    answer_dict = list(map(lambda x: _filter_label(x), answer_dict))
    answer_dict = list(filter(None, answer_dict))
    return answer_dict, language

