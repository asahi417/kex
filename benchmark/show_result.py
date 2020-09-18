import json
from glob import glob

table = "|         Model         |    F1 (P/R) @5    |    F1 (P/R) @10   |     F1 (P/R) @15    | approx time (sec) |\n"
table += "|:---------------------:|:-----------------:|:-----------------:|:-------------------:|:-----------------:|\n"
d = 3
for i in glob('./benchmark/*.json'):
    tmp = json.load(open(i))
    algo = i.split('.')[-2]
    algo = algo.replace('rank', 'Rank').replace('tfidf', 'TFIDF').replace('expand', 'Expand').\
        replace('text', 'Text').replace('topic', 'Topic').replace('single', 'Single').replace('page', 'Page').\
        replace('multi', 'Multi').replace('position', 'Position')

    row = "| {0} |".format(algo)
    row += " {0} ({1}/{2}) |".format(
        round(tmp['top_5']['f_1'], d),
        round(tmp['top_5']['mean_precision'], d),
        round(tmp['top_5']['mean_recall'], d))
    row += " {0} ({1}/{2}) |".format(
        round(tmp['top_10']['f_1'], d),
        round(tmp['top_10']['mean_precision'], d),
        round(tmp['top_10']['mean_recall'], d))
    row += " {0} ({1}/{2}) |".format(
        round(tmp['top_15']['f_1'], d),
        round(tmp['top_15']['mean_precision'], d),
        round(tmp['top_15']['mean_recall'], d))
    row += " {0}|\n".format(round(tmp['process_time_second'], d))
    table += row

print(table)
