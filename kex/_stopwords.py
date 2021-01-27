""" stopwords are fetched from https://github.com/LIAAD/yake/tree/master/yake/StopwordsList """
import os
from typing import List


def get_stopwords_list(language: str = 'en', stopwords_list: List = None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    local_path = os.path.join("_stopwords", "stopwords_{}.txt".format(language[:2]))
    if not os.path.exists(os.path.join(dir_path, local_path)):
        local_path = os.path.join("StopwordsList", "stopwords_noLang.txt")

    resource_path = os.path.join(dir_path, local_path)
    try:
        with open(resource_path, encoding='utf-8') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")
    except UnicodeEncodeError:
        with open(resource_path, encoding='ISO-8859-1') as stop_fil:
            _stopwords_list = stop_fil.read().lower().split("\n")

    if stopwords_list:
        return list(set(_stopwords_list + stopwords_list))
    else:
        return list(set(_stopwords_list))
