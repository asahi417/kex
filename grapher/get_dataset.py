import os
from glob import glob

VALID_DATASET_LIST = [
    "110-PT-BN-KP", "WikiNews",
    "cacic", "citeulike180", "fao30", "fao780", "Inspec", "kdd",
    "Krapivin2009", "Nguyen2007", "pak2018", "PubMed", "Schutz2008", "SemEval2010", "SemEval2017", "theses100", "wicc",
    "wiki20", "www", "500N-KPCrowd-v1.1"
]

__all__ = ('get_benchmark_dataset', "VALID_DATASET_LIST")


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
    answer_dict = [{
        "keywords": open(
            "{}/{}/keys/{}.key".format(
                cache_dir, data, '.txt'.join(os.path.basename(t).split('.txt')[:-1])),
            'r').read().split('\n'),
        "source": open(t, 'r').read(),
        "id": os.path.basename(t)} for t in glob("{}/{}/docsutf8/*.txt".format(cache_dir, data))]
    language = open("{}/{}/language.txt".format(cache_dir, data)).read()
    return answer_dict, language
