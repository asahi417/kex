import pandas as pd
from glob import glob

for i in glob('benchmark/*/prediction.*.csv'):
    df_prediction_cached = pd.read_csv(i, index_col=0)
    labels = [i.split('||') if type(i) is str else '' for i in df_prediction_cached['label'].values.tolist()]
    preds = [i.split('||') if type(i) is str else '' for i in df_prediction_cached['label_predict'].values.tolist()]

    # run algorithm and test it over data
    mrr = []
    pre5 = []
    for n_, (l_, p) in enumerate(zip(labels, preds)):
        all_ranks = [n + 1 for n, p_ in enumerate(p) if p_ in l_]
        if len(all_ranks) == 0:  # no answer is found (TopicRank may cause it)
            mrr.append(1 / (len(preds) + 1))
        else:
            mrr.append(1 / min(all_ranks))
        n = min(5, len(l_))
        positive = list(set(p[:n]).intersection(set(l_)))
        pre5.append(len(positive) / n)
    df_prediction_cached['mrr'] = mrr
    df_prediction_cached['pre_5'] = pre5
    df_prediction_cached.to_csv(i)
    print(df_prediction_cached.head())
