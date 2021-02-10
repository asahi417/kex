""" stopwords are fetched from https://github.com/LIAAD/yake/tree/master/yake/StopwordsList """
import os
import requests
import tarfile
import logging
from typing import List
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

CACHE_DIR = '{}/cache_kex'.format(os.path.expanduser('~'))
STOPWORDS_DIR = '{}/stopwords'.format(CACHE_DIR)


def download_stopwords_dump():
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = "https://github.com/asahi417/kex/raw/master/asset/stopwords.tar.gz"
    filename = '{}/{}'.format(CACHE_DIR, os.path.basename(url))
    logging.info('downloading stopwords dump from `{}` to `{}`'.format(url, CACHE_DIR))
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(CACHE_DIR)
    tar.close()


def get_stopwords_list(language: str = 'en', stopwords_list: List = None):
    """ Built-in stopwords list

     Parameter
    ------------------
    language: str
    stopwords_list: List
        A custom stopword list to concatenate to add to built-in stopwords

     Return
    ------------------
    A list of stopwords
    """
    if not os.path.exists(STOPWORDS_DIR):
        download_stopwords_dump()

    local_path = os.path.join(CACHE_DIR, "stopwords/stopwords_{}.txt".format(language[:2]))

    if not os.path.exists(os.path.join(CACHE_DIR, local_path)):
        local_path = os.path.join(CACHE_DIR, "stopwords/stopwords_noLang.txt")

    try:
        with open(local_path, encoding='utf-8') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")
    except UnicodeEncodeError:
        with open(local_path, encoding='ISO-8859-1') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")
    except FileNotFoundError:
        logging.exception('skip loading stopwords due to the error')
        _stopwords_list = []

    if stopwords_list:
        return list(set(_stopwords_list + stopwords_list))
    else:
        return list(set(_stopwords_list))
