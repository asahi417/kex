# Grapher: graph-based keyword extraction
*Grapher*, a python library of modern unsurpervised keyword extraction algorithms based on graph approach.

## Get Started
Clone this repository and pip install

```
git clone https://github.com/asahi417/
cd grapher
pip install .
```

### Basic Usage
Small example to use **keyphraser**: 

```python
import graph_keyword_extractor
model = graph_keyword_extractor.graph.TopicRank()
sample = [
'''
We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier detection.
It starts by training word embeddings on the target document to capture semantic regularities among the words. It then
uses the minimum covariance determinant estimator to model the distribution of non-keyphrase word vectors, under the
assumption that these vectors come from the same distribution, indicative of their irrelevance to the semantics
expressed by the dimensions of the learned vector representation. Candidate keyphrases only consist of words that are
detected as outliers of this dominant distribution. Empirical results show that our approach outperforms state
of-the-art and recent unsupervised keyphrase extraction methods.
''',
'''
Keyphrase extraction is the task of automatically selecting a small set of phrases that
best describe a given free text document. Supervised keyphrase extraction requires large
amounts of labeled training data and generalizes very poorly outside the domain of the
training data. At the same time, unsupervised systems have poor accuracy, and often
do not generalize well, as they require the input document to belong to a larger corpus also
given as input. Addressing these drawbacks,
in this paper, we tackle keyphrase extraction from single documents with EmbedRank:
a novel unsupervised method, that leverages
sentence embeddings. EmbedRank achieves
higher F-scores than graph-based state of the
art systems on standard datasets and is suitable for real-time processing of large amounts
of Web data. With EmbedRank, we also explicitly increase coverage and diversity among
the selected keyphrases by introducing an
embedding-based maximal marginal relevance
(MMR) for new phrases. A user study including over 200 votes showed that,
although reducing the phrases’ semantic overlap leads to
no gains in F-score, our high diversity selection is preferred by humans.
''']
phrases = model.extract(sample, count=2)
```

The output `phrases` has data structure corresponding to the given document, which consists of
*stemmed phrase*, *score*, and *lexical information* as below. Those are in order of *score*.

```
[
    [
        ('novel unsupervis keyphras extract approach',
         0.21610061365023375,
         {
            'stemmed': 'novel unsupervis keyphras extract approach',
            'pos': 'ADJ ADJ NOUN NOUN NOUN',
            'raw': ['novel unsupervised keyphrase extraction approach'],
            'lemma': ['novel unsupervised keyphrase extraction approach'],
            'offset': [[3, 7]],
            'count': 1,
            'original_sentence_token_size': 54
         }),
         ('keyphras word vector',
          0.1759375790487177,
          {
            'stemmed': 'keyphras word vector',
            'pos': 'NOUN NOUN NOUN',
            'raw': ['keyphrase word vectors'],
            'lemma': ['keyphrase word vector'],
            'offset': [[49, 51]],
            'count': 1,
            'original_sentence_token_size': 54
          })
    ],
    [
        ('free text document',
         0.10465865367621371,
         {
            'stemmed': 'free text document',
            'pos': 'ADJ NOUN NOUN',
            'raw': ['free text document'],
            'lemma': ['free text document'],
            'offset': [[18, 20]],
            'count': 1,
            'original_sentence_token_size': 54
         }),
        ('input document',
         0.09090852474669106,
         {
            'stemmed': 'input document',
            'pos': 'NOUN NOUN',
            'raw': ['input document'],
            'lemma': ['input document'],
            'offset': [[66, 67]],
            'count': 1,
            'original_sentence_token_size': 54
         })
    ]
]
```


## List of Algorithms

- Statistical Approaches
    - TFIDF Based         [[implementation](graph_keyword_extractor/algorithms/stat/tfidf_based.py), [paper](http://aclweb.org/anthology/S10-1041)]
    - **RAKE (WIP)**      [[implementation](graph_keyword_extractor/algorithms/stat/rake.py),        [paper](https://pdfs.semanticscholar.org/5a58/00deb6461b3d022c8465e5286908de9f8d4e.pdf)]
    - **EmbedRank (WIP)** [[implementation](graph_keyword_extractor/algorithms/stat/embed_rank.py),  [paper](http://www.aclweb.org/anthology/K18-1022)] 

- Graph-based Approaches
    - Basic Algorithms
        - TextRank      [[implementation](graph_keyword_extractor/algorithms/graph/text_rank.py),   [paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)]
        - SingleRank    [[implementation](graph_keyword_extractor/algorithms/graph/single_rank.py), [paper](https://aclanthology.info/pdf/C/C08/C08-1122.pdf)]
        - ExpandRank    [[implementation](graph_keyword_extractor/algorithms/graph/expand_rank.py), [paper](https://pdfs.semanticscholar.org/8a99/634e0b418ee61c9bd81f61d334b80486dc53.pdf)]
        - PositionRank  [[implementation](graph_keyword_extractor/algorithms/graph/expand_rank.py), [paper](http://people.cs.ksu.edu/~ccaragea/papers/acl17.pdf)]
    - Topic-Aware
        - TopicalPageRank   [[implementation](graph_keyword_extractor/algorithms/graph/topical_page_rank.py),        [paper](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf)]
        - SingleTPR         [[implementation](graph_keyword_extractor/algorithms/graph/single_topical_page_rank.py), [paper](https://core.ac.uk/download/pdf/55828317.pdf)]
        - TopicRank         [[implementation](graph_keyword_extractor/algorithms/graph/topic_rank.py),               [paper](http://www.aclweb.org/anthology/I13-1062)]
        - MultipartiteRank  [[implementation](graph_keyword_extractor/algorithms/graph/multipartite_rank.py),        [paper](https://arxiv.org/pdf/1803.08721.pdf) ] 