import os
import re
from glob import glob

from ._phrase_constructor import PhraseConstructor
from nltk.stem.porter import PorterStemmer  # only for English

STEMMER = PorterStemmer()
VALID_DATASET_LIST = [
    # "110-PT-BN-KP", "cacic", "pak2018", "WikiNews", "wicc"
    "citeulike180", "fao30", "fao780", "Inspec", "kdd", "Krapivin2009", "Nguyen2007", "PubMed", "Schutz2008",
    "SemEval2010", "SemEval2017", "theses100", "wiki20", "www", "500N-KPCrowd-v1.1"
]

__all__ = ('get_benchmark_dataset', "VALID_DATASET_LIST", "get_statistics")


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

    return list(filter(
        lambda x: len(x) > 0, [cleaner(s) for s in re.split('[\n,]', string)])
    )


def get_benchmark_dataset(data: str = 'Inspec', cache_dir: str = "./cache"):
    """ to get a dataset for keyword extraction (all not stemmed)

     Parameter
    -------------
    data: str
        dataset name
    cache_dir: str
        directory to cache the data

     Return
    -------------
    dictionary consists of the data
    language
    """
    assert data in VALID_DATASET_LIST, "undefined dataset: {}".format(data)
    url = "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists("{}/{}".format(cache_dir, data)):
        os.system('wget -O {0}/{2}.zip {1}/{2}.zip'.format(cache_dir, url, data))
        if data == "110-PT-BN-KP":
            os.system("unzip {0}/{1}.zip -d {0}/{1}".format(cache_dir, data))
        elif data == "WikiNews":
            os.system("unzip {0}/{1}.zip -d {0}".format(cache_dir, data))
            os.system("mv {0}/WKC {0}/{1}".format(cache_dir, data))
        else:
            os.system("unzip {0}/{1}.zip -d {0}/".format(cache_dir, data))
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
    return answer_dict, language


def get_statistics(keywords, source):
    """Data level feature (per entry):
    - # gold label
    - # word (raw/no stopword)
    - # candidate
    - # unique word (raw/no stopword)
    - # mean/std of word distribution (raw/no stopword)
    """
    phraser = PhraseConstructor()
    phrase, stemmed_token = phraser.tokenize_and_stem_and_phrase(source)
    keywords_valid = list(set(phrase.keys()).intersection(set(keywords)))
    keywords_invalid = list(set(keywords) - set(keywords_valid))
    stemmed_text = ' '.join(stemmed_token)
    keywords_invalid_appeared = list(filter(lambda x: x in stemmed_text, keywords_invalid))
    keywords_invalid_intractable = list(set(keywords_invalid) - set(keywords_invalid_appeared))

    out = {
        "n_phrase": len(phrase),
        "n_label": len(keywords),
        "n_label_in_candidates": len(keywords_valid),
        "n_label_out_candidates": len(keywords_invalid_appeared),
        "n_label_intractable": len(keywords_invalid_intractable),
        "label_in_candidates": keywords_valid,
        "label_out_candidates": keywords_invalid_appeared,
        "label_intractable": keywords_invalid_intractable
    }

    def _tmp(i):
        sufix = '' if i else '_with_stopword'
        tokens = phraser.tokenize_and_stem(source, apply_stopwords=i)
        dist = list(map(lambda x: sum(map(lambda y: y == x, tokens)), set(tokens)))
        mean = sum(dist) / len(dist)
        return {
            'n_word{}'.format(sufix): len(tokens),
            'n_unique_word{}'.format(sufix): len(dist),
            'mean{}'.format(sufix): mean,
            'std{}'.format(sufix): (sum(map(lambda x: (x - mean) ** 2, dist)) / len(dist)) ** 0.5
        }

    dicts = [_tmp(_i) for _i in [True, False]]
    out.update(dicts[0])
    out.update(dicts[1])
    return out
