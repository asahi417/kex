""" stopwords are fetched from https://github.com/LIAAD/yake/tree/master/yake/StopwordsList """
import os
from typing import List

CACHE_DIR = '{}/cache_kex'.format(os.path.expanduser('~'))
URL = "https://github.com/asahi417/kex/raw/master/asset/stopwords.tar.gz"
if not os.path.exists('{}/stopwords.tar.gz'.format(CACHE_DIR)):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.system('wget -O {0}/stopwords.tar.gz {1}'.format(CACHE_DIR, URL))
    os.system('tar xf {0}/stopwords.tar.gz -C {0}'.format(CACHE_DIR))


def get_stopwords_list(language: str = 'en', stopwords_list: List = None):
    local_path = os.path.join(CACHE_DIR, "stopwords/stopwords_{}.txt".format(language[:2]))

    if not os.path.exists(os.path.join(CACHE_DIR, local_path)):
        local_path = os.path.join(CACHE_DIR, "stopwords/stopwords_noLang.txt")

    try:
        with open(local_path, encoding='utf-8') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")
    except UnicodeEncodeError:
        with open(local_path, encoding='ISO-8859-1') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")

    if stopwords_list:
        return list(set(_stopwords_list + stopwords_list))
    else:
        return list(set(_stopwords_list))
