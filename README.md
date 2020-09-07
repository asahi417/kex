
# Grapher: graph-based keyword extraction


<p align="center">
  <img src="./asset/topic_rank_fig.png" width="800">
  <br><i>Fig 1: Graph created by TopicRank (Bougouin et al.,13) </i>
</p>


*Grapher*, a python library of modern unsurpervised keyword extraction algorithms based on graph approach.

## Get Started
Clone this repository and pip install

```
git clone https://github.com/asahi417/grapher
cd grapher
pip install .
```

## Basic Usage
```python
>>> import grapher
>>> model = grapher.models.TopicRank()
>>> sample = '''
We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier detection.
It starts by training word embeddings on the target document to capture semantic regularities among the words. It then
uses the minimum covariance determinant estimator to model the distribution of non-keyphrase word vectors, under the
assumption that these vectors come from the same distribution, indicative of their irrelevance to the semantics
expressed by the dimensions of the learned vector representation. Candidate keyphrases only consist of words that are
detected as outliers of this dominant distribution. Empirical results show that our approach outperforms state
of-the-art and recent unsupervised keyphrase extraction methods.
'''
>>> model.get_keywords(sample)

[{'count': 1,
  'lemma': 'word embedding',
  'n_source_tokens': 120,
  'offset': [[22, 23]],
  'pos': 'NOUN NOUN',
  'raw': 'word embeddings',
  'score': 0.13759301735957652,
  'stemmed': 'word embed'},
 {'count': 1,
  'lemma': 'novel unsupervised keyphrase extraction approach',
  'n_source_tokens': 120,
  'offset': [[4, 8]],
  'pos': 'ADJ ADJ NOUN NOUN NOUN',
  'raw': 'novel unsupervised keyphrase extraction approach',
  'score': 0.13064559025892963,
  'stemmed': 'novel unsupervis keyphras extract approach'},
 {'count': 1,
  'lemma': 'distribution',
  'n_source_tokens': 120,
  'offset': [[48, 49]],
  'pos': 'NOUN',
  'raw': 'distribution',
  'score': 0.1042586518389951,
  'stemmed': 'distribut'},
 {'count': 1,
  'lemma': 'minimum covariance determinant estimator',
  'n_source_tokens': 120,
  'offset': [[41, 44]],
  'pos': 'ADJ NOUN NOUN NOUN',
  'raw': 'minimum covariance determinant estimator',
  'score': 0.07845448951551706,
  'stemmed': 'minimum covari determin estim'}]
```

## Algorithms
- Graph-based Approaches
    - Basic Algorithms
        - [TextRank, Mihalcea et al., 04](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
        - [SingleRank, Wan et al., 08](https://aclanthology.info/pdf/C/C08/C08-1122.pdf)]
        - [ExpandRank, Wan et al., 08](https://pdfs.semanticscholar.org/8a99/634e0b418ee61c9bd81f61d334b80486dc53.pdf)]
        - [PositionRank, Florescu et al.,18](http://people.cs.ksu.edu/~ccaragea/papers/acl17.pdf)]
    - Topic-Aware
        - [TopicalPageRank, Liu et al.,10](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf)]
        - [SingleTPR, Sterckx et al.,15](https://core.ac.uk/download/pdf/55828317.pdf)]
        - [TopicRank, Bougouin et al.,13](http://www.aclweb.org/anthology/I13-1062)]
        - [MultipartiteRank, Boudin 18](https://arxiv.org/pdf/1803.08721.pdf) ]

- Statistical Approaches
    - [TFIDF Based](http://aclweb.org/anthology/S10-1041)]
    - [RAKE (WIP)](https://pdfs.semanticscholar.org/5a58/00deb6461b3d022c8465e5286908de9f8d4e.pdf)]
    - [EmbedRank (WIP)](http://www.aclweb.org/anthology/K18-1022)] 

 