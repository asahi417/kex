# """ Get statistics of kex's build in dataset """
# import argparse
# import os
# import logging
# from tqdm import tqdm
# from itertools import chain
#
# import pandas as pd
# import kex
#
#
# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')  # should be right after import logging
#
# types = {
#     "Inspec": "Abst",
#     "www": "Abst",
#     "kdd": "Abst",
#     "Krapivin2009": "Full",
#     "SemEval2010": "Full",
#     "SemEval2017": "Para",
#     "citeulike180": "Full",
#     "PubMed": "Full",
#     "Schutz2008": "Full",
#     "theses100": "Full",
#     "fao30": "Full",
#     "fao780": "Full",
#     "Nguyen2007": "Full",
#     "wiki20": "Report",
#     "500N-KPCrowd-v1.1": "News"
# }
# domain = {
#     "Inspec": "CS",
#     "www": "CS",
#     "kdd": "CS",
#     "Krapivin2009": "CS",
#     "SemEval2010": "CS",
#     "SemEval2017": "-",
#     "citeulike180": "BI",
#     "PubMed": "BM",
#     "Schutz2008": "BM",
#     "theses100": "-",
#     "fao30": "AG",
#     "fao780": "AG",
#     "Nguyen2007": "-",
#     "wiki20": "CS",
#     "500N-KPCrowd-v1.1": "-"
# }
# # BI: bioinfomatics
# # BM biomedical
# # AD: agricultural document
# order = ["500N-KPCrowd-v1.1", 'Inspec', 'Krapivin2009', 'Nguyen2007', 'PubMed', 'Schutz2008', 'SemEval2010', 'SemEval2017',
#          'citeulike180', 'fao30', 'fao780', 'theses100', 'kdd', 'wiki20', 'www']
#
# def get_statistics(keywords, source: str):
#     """ Data level feature (per entry):
#     {
#         "n_phrase":
#         "n_label":
#         "n_label_in_candidates":
#         "n_label_out_candidates":
#         "n_label_intractable":
#         "label_in_candidates":
#         "label_out_candidates":
#         "label_intractable":
#         "n_word"
#         "n_unique_word":
#         "mean":
#         "std":
#         "n_word_with_stopword"
#         "n_unique_word_with_stopword":
#         "mean_with_stopword":
#         "std_with_stopword":
#     }
#
#      Parameter
#     -------------
#     keywords: list
#         a list of keywords
#     source: str
#         directory to cache the data
#
#      Return
#     -------------
#     A dictionary containing the above statistics
#     """
#     phraser = kex.PhraseConstructor()
#     phrase, stemmed_token = phraser.tokenize_and_stem_and_phrase(source)
#     keywords_valid = list(set(phrase.keys()).intersection(set(keywords)))
#     keywords_invalid = list(set(keywords) - set(keywords_valid))
#     stemmed_text = ' '.join(stemmed_token)
#     keywords_invalid_appeared = list(filter(lambda x: x in stemmed_text, keywords_invalid))
#     keywords_invalid_intractable = list(set(keywords_invalid) - set(keywords_invalid_appeared))
#
#     out = {
#         "n_phrase": len(phrase),
#         "n_label": len(keywords),
#         "n_label_in_candidates": len(keywords_valid),
#         "n_label_out_candidates": len(keywords_invalid_appeared),
#         "n_label_intractable": len(keywords_invalid_intractable),
#         "label_in_candidates": keywords_valid,
#         "label_out_candidates": keywords_invalid_appeared,
#         "label_intractable": keywords_invalid_intractable
#     }
#
#     def _tmp(i):
#         sufix = '' if i else '_with_stopword'
#         tokens = phraser.tokenize_and_stem(source, apply_stopwords=i)
#         dist = list(map(lambda x: sum(map(lambda y: y == x, tokens)), set(tokens)))
#         mean = sum(dist) / len(dist)
#         return {
#             'n_word{}'.format(sufix): len(tokens),
#             'n_unique_word{}'.format(sufix): len(dist),
#             'mean{}'.format(sufix): mean,
#             'std{}'.format(sufix): (sum(map(lambda x: (x - mean) ** 2, dist)) / len(dist)) ** 0.5
#         }
#
#     dicts = [_tmp(_i) for _i in [True, False]]
#     out.update(dicts[0])
#     out.update(dicts[1])
#     return out
#
#
# def get_options():
#     parser = argparse.ArgumentParser(description='Benchmark preset methods in kex')
#     parser.add_argument('-d', '--data', help='data:{}'.format(kex.VALID_DATASET_LIST), default=None, type=str)
#     parser.add_argument('-e', '--export', help='log export dir', default='./benchmark', type=str)
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     opt = get_options()
#     data_list = kex.VALID_DATASET_LIST if opt.data is None else opt.data.split(',')
#     total_dict = {}
#     for data_n, data in enumerate(data_list):
#         logging.info('data: {}'.format(data))
#         logging.info('computing stats on each data')
#         _file = '{}/{}/statistics.csv'.format(opt.export, data)
#         if os.path.exists(_file):
#             logging.info(' - skip statistics computation: statistics file found at {}'.format(_file))
#             df_stats = pd.read_csv(_file, index_col=0)
#         else:
#             stats = [
#                 'filename', 'n_phrase',
#                 'n_label', 'n_label_in_candidates', 'n_label_out_candidates', 'n_label_intractable',
#                 'n_word', 'n_unique_word', 'mean', 'std',
#                 'n_word_with_stopword', 'n_unique_word_with_stopword', 'mean_with_stopword', 'std_with_stopword',
#                 'label_in_candidates', 'label_out_candidates', 'label_intractable'
#             ]
#             os.makedirs(os.path.join(opt.export, data), exist_ok=True)
#             df_stats = pd.DataFrame(index=stats)
#             tmp, language = kex.get_benchmark_dataset(data, keep_only_valid_label=False)
#             for n, i in enumerate(tqdm(tmp)):
#                 keywords_, source_, filename = i['keywords'], i['source'], i['id']
#                 out = get_statistics(keywords_, source_)
#                 out['label_in_candidates'] = '||'.join(out['label_in_candidates'])
#                 out['label_out_candidates'] = '||'.join(out['label_out_candidates'])
#                 out['label_intractable'] = '||'.join(out['label_intractable'])
#                 out['filename'] = filename
#                 df_stats[n] = [out[i] for i in stats]
#             df_stats = df_stats.T
#             df_stats.to_csv(_file)
#
#         tmp = len(list(chain(*[
#             [k for k in i.split('||') if len(k.split(' ')) > 1]
#             for i in df_stats['label_in_candidates'].values.tolist() if type(i) is str])))
#
#         dist = list(map(lambda x: sum(map(lambda y: y == x, df_stats["n_phrase"])), set(df_stats["n_phrase"])))
#         mean = sum(dist) / len(dist)
#         (sum(map(lambda x: (x - mean) ** 2, dist)) / len(dist)) ** 0.5
#
#         total_dict[data] = {
#             "Domain": domain[data],
#             "Type": types[data],
#             "Data size": len(df_stats),
#             "Avg word": df_stats["n_word"].mean(),
#             "Avg vocab": df_stats["n_unique_word"].mean(),
#             "Avg keyword": df_stats["n_label"].mean(),
#             "Avg keyword (in candidate)": df_stats["n_label_in_candidates"].mean(),
#             "Avg keyword (in candidate and multiwords)": tmp/len(df_stats),
#             "Vocab diversity": df_stats["n_unique_word"].sum()/df_stats["n_word"].sum(),
#             "Avg phrase": df_stats["n_phrase"].sum()/len(df_stats),
#             "Std phrases":
# 	        "Std word":
#             "Std vocab":
#             "Std keyword (tractable)":
#         }
#     df = pd.DataFrame(total_dict).T
#     df.to_csv('{}/data_statistics.csv'.format(opt.export))
