import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_mrr = pd.read_csv('corrMat_mrr.csv', index_col=0)
df_pre5 = pd.read_csv('corrMat_pre5.csv', index_col=0)
export_dir_root = '.'
order = ['FirstN', 'TF', 'LexSpec', 'TFIDF', 'TextRank', 'SingleRank', 'PositionRank', 'LexRank', 'TFIDFRank',
         'SingleTPR', 'TopicRank']


def heatmap():
    # plot heatmap
    for name, df in zip(['heatmap_mrr', 'heatmap_pre5'], [df_mrr, df_pre5]):
        fig = plt.figure()
        fig.clear()
        df = df.astype(float).round(2).T[order].T
        # df = df *
        sns_plot = sns.heatmap(df, annot=True, fmt="g", cbar=True)
        sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=60)
        sns_plot.tick_params(labelsize=10)
        fig = sns_plot.get_figure()
        plt.tight_layout()
        fig.savefig('{}/{}.png'.format(export_dir_root, name))
        fig.savefig('{}/{}.pdf'.format(export_dir_root, name))


if __name__ == '__main__':
    heatmap()
