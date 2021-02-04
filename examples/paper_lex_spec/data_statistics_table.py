import os
import logging
from tqdm import tqdm
from itertools import chain

import pandas as pd
import kex

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

types = {
    "Inspec": "Abst",
    "www": "Abst",
    "kdd": "Abst",
    "Krapivin2009": "Full",
    "SemEval2010": "Full",
    "SemEval2017": "Para",
    "citeulike180": "Full",
    "PubMed": "Full",
    "Schutz2008": "Full",
    "theses100": "Full",
    "fao30": "Full",
    "fao780": "Full",
    "Nguyen2007": "Full",
    "wiki20": "Report",
    "500N-KPCrowd-v1.1": "News"
}
domain = {
    "Inspec": "CS",
    "www": "CS",
    "kdd": "CS",
    "Krapivin2009": "CS",
    "SemEval2010": "CS",
    "SemEval2017": "-",
    "citeulike180": "BI",
    "PubMed": "BM",
    "Schutz2008": "BM",
    "theses100": "-",
    "fao30": "AG",
    "fao780": "AG",
    "Nguyen2007": "-",
    "wiki20": "CS",
    "500N-KPCrowd-v1.1": "-"
}
# BI: bioinfomatics
# BM biomedical
# AD: agricultural document
data_list = ["500N-KPCrowd-v1.1", 'Inspec', 'Krapivin2009', 'Nguyen2007', 'PubMed', 'Schutz2008', 'SemEval2010',
             'SemEval2017', 'citeulike180', 'fao30', 'fao780', 'theses100', 'kdd', 'wiki20', 'www']
phraser = kex.PhraseConstructor()


def get_statistics(data: str):
    """ get statistics"""
    word = []
    n_vocab = []
    n_word = []
    n_phrase = []
    n_label = []
    n_label_in_candidates = []
    n_label_in_candidates_multi = []
    label_in_candidates = []

    dataset, language = kex.get_benchmark_dataset(data, keep_only_valid_label=False)
    if language == 'en':
        return
    output = {'Data size': len(dataset), "Domain": domain[data], "Type": types[data]}

    for data in tqdm(dataset):
        phrase, stemmed_token = phraser.tokenize_and_stem_and_phrase(data['source'])
        keywords_valid = list(set(phrase.keys()).intersection(set(data['keywords'])))
        word.append(stemmed_token)
        n_vocab.append(len(list(set(stemmed_token))))
        n_word.append(len(stemmed_token))
        n_phrase.append(len(phrase))
        n_label.append(len(data['keywords']))
        n_label_in_candidates.append(len(keywords_valid))
        n_label_in_candidates_multi.append(len([k for k in keywords_valid if len(k.split(' ')) > 1]))
        label_in_candidates.append(keywords_valid)


    output['Avg phrase'] = sum(n_phrase) / len(dataset)
    output['Std phrase'] = (sum([(a - output['Avg phrase']) ** 2 for a in n_phrase]) / len(dataset)) ** 0.5

    output['Avg word'] = sum(n_word) / len(dataset)
    output['Std word'] = (sum([(a - output['Avg word']) ** 2 for a in n_word]) / len(dataset)) ** 0.5

    output['Avg vocab'] = sum(n_vocab) / len(dataset)
    output['Std vocab'] = (sum([(a - output['Avg vocab']) ** 2 for a in n_vocab]) / len(dataset)) ** 0.5

    output['Avg keyword'] = sum(n_label) / len(dataset)
    output['Std vocab'] = (sum([(a - output['Avg vocab']) ** 2 for a in n_vocab]) / len(dataset)) ** 0.5

    output['Avg keyword (in candidate)'] = sum(n_label_in_candidates) / len(dataset)
    output['Std keyword (in candidate)'] = (sum([(a - output['Avg keyword (in candidate)']) ** 2 for a in n_label_in_candidates]) / len(dataset)) ** 0.5

    output['Avg keyword (in candidate & multi)'] = sum(n_label_in_candidates_multi) / len(dataset)
    output['Std keyword (in candidate & multi)'] = (sum([(a - output['Avg keyword (in candidate & multi)']) ** 2 for a in n_label_in_candidates_multi]) / len(dataset)) ** 0.5

    output['Vocab diversity'] = sum(n_word) / sum(n_vocab)

    return output


if __name__ == '__main__':

    all_stats = {}
    for data_name in data_list:
        logging.info('data: {}'.format(data_name))
        all_stats[data_name] = get_statistics(data_name)

    df = pd.DataFrame(all_stats)
    print(df)
    df.to_csv('./benchmark/data_statistics.csv')
