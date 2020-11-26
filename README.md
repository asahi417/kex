# Grapher: Graph-based Keyword Extractions

<p align="center">
  <img src="./asset/topic_rank_fig.png" width="800">
  <br><i>Fig 1: TopicRank visualization (Bougouin et al.,13) </i>
</p>


*Grapher*, a quick python library to work on graph-based unsurpervised keyword extraction algorithms.


## Get Started
Install via pip
```
pip install git+https://github.com/asahi417/grapher
```

or clone and install

```shell script
git clone https://github.com/asahi417/grapher
cd grapher
pip install .
```

## Basic Usage
*Grapher* retrieves keywords from a document with various graph-based algorithms:
- `TFIDF`: a simple statistic baseline
- `TextRank`: [Mihalcea et al., 04](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
- `SingleRank`: [Wan et al., 08](https://aclanthology.info/pdf/C/C08/C08-1122.pdf)
- `TopicRank`: [Bougouin et al.,13](http://www.aclweb.org/anthology/I13-1062)
- `PositionRank`: [Florescu et al.,18](http://people.cs.ksu.edu/~ccaragea/papers/acl17.pdf)
- `MultipartiteRank`: [Boudin 18](https://arxiv.org/pdf/1803.08721.pdf)
- `ExpandRank`: [Wan et al., 08](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf)
- `TopicalPageRank`: [Liu et al.,10](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf)
- `SingleTPR`: [Sterckx et al.,15](https://core.ac.uk/download/pdf/55828317.pdf)

All the algorithms can be simply used as below.

```python
>>> import grapher
>>> model = grapher.PositionRank()  # any algorithm listed above
>>> sample = '''
We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier detection.
It starts by training word embeddings on the target document to capture semantic regularities among the words. It then
uses the minimum covariance determinant estimator to model the distribution of non-keyphrase word vectors, under the
assumption that these vectors come from the same distribution, indicative of their irrelevance to the semantics
expressed by the dimensions of the learned vector representation. Candidate keyphrases only consist of words that are
detected as outliers of this dominant distribution. Empirical results show that our approach outperforms state
of-the-art and recent unsupervised keyphrase extraction methods.
'''
>>> model.get_keywords(sample, n_keywords=2)
[
 {'count': 1,
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
  'stemmed': 'novel unsupervis keyphras extract approach'}
]
```

### Algorithm with prior
Algorithms with priors need to be trained beforehand (`TFIDF`, `ExpandRank`, `TopicalPageRank`, `SingleTPR`)
```python
>>> import grapher
>>> model = grapher.SingleTPR()
>>> test_sentences = ['documentA', 'documentB', 'documentC']
>>> model.train(test_sentences, export_directory='./tmp')
``` 

Priors are cached and can be loaded on the fly.

```python
>>> import grapher
>>> model = grapher.SingleTPR()
>>> model.load('./tmp')
```

### Supported Language
Currently, all the algorithms are available in English, but soon will relax the constrain to allow other language support.
The dependency is mainly due to the PoS tagger and the word stemmer.

## Benchamrk

We 

|dataset     |singlerank         |expandrank         |positionrank       |singletpr          |textrank           |lexrank            |lexspec            |topicrank          |tfidf              |
|------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|pubmed      |0.049 (0.066/0.039)|0.038 (0.052/0.031)|0.046 (0.062/0.037)|0.044 (0.059/0.035)|0.051 (0.069/0.041)|0.039 (0.053/0.031)|0.033 (0.044/0.026)|0.032 (0.043/0.026)|0.028 (0.038/0.022)|
|inspec      |0.195 (0.235/0.166)|0.184 (0.222/0.158)|0.179 (0.216/0.153)|0.167 (0.201/0.142)|0.194 (0.233/0.165)|0.182 (0.219/0.155)|0.169 (0.204/0.144)|0.151 (0.182/0.129)|0.171 (0.206/0.146)|
|nguyen2007  |0.094 (0.103/0.086)|0.101 (0.111/0.093)|0.121 (0.133/0.111)|0.085 (0.093/0.078)|0.07 (0.078/0.065) |0.104 (0.114/0.095)|0.093 (0.102/0.085)|0.083 (0.091/0.076)|0.082 (0.09/0.075) |
|wiki20      |0.048 (0.11/0.031) |0.057 (0.13/0.037) |0.046 (0.105/0.03) |0.051 (0.115/0.032)|0.026 (0.06/0.017) |0.053 (0.12/0.034) |0.044 (0.1/0.028)  |0.051 (0.115/0.032)|0.051 (0.115/0.032)|
|krapivin2009|0.076 (0.058/0.109)|0.076 (0.059/0.11) |0.114 (0.088/0.164)|0.073 (0.056/0.105)|0.053 (0.041/0.077)|0.077 (0.059/0.11) |0.065 (0.05/0.094) |0.065 (0.05/0.093) |0.056 (0.043/0.081)|
|kdd         |0.046 (0.032/0.079)|0.048 (0.034/0.082)|0.048 (0.034/0.083)|0.043 (0.03/0.074) |0.044 (0.031/0.076)|0.048 (0.034/0.082)|0.045 (0.032/0.077)|0.033 (0.023/0.057)|0.047 (0.033/0.081)|
|semeval2010 |0.08 (0.102/0.066) |0.077 (0.099/0.063)|0.112 (0.143/0.092)|0.072 (0.092/0.059)|0.06 (0.077/0.049) |0.08 (0.103/0.066) |0.067 (0.086/0.055)|0.083 (0.106/0.068)|0.064 (0.082/0.053)|
|fao780      |0.088 (0.079/0.099)|0.083 (0.074/0.093)|0.082 (0.073/0.092)|0.089 (0.08/0.1)   |0.081 (0.072/0.091)|0.083 (0.074/0.093)|0.071 (0.064/0.08) |0.054 (0.049/0.061)|0.061 (0.055/0.069)|
|fao30       |0.098 (0.207/0.064)|0.082 (0.173/0.054)|0.09 (0.19/0.059)  |0.101 (0.213/0.066)|0.087 (0.183/0.057)|0.084 (0.177/0.055)|0.065 (0.137/0.042)|0.047 (0.1/0.031)  |0.057 (0.12/0.037) |
|semeval2017 |0.23 (0.317/0.18)  |0.227 (0.314/0.178)|0.214 (0.295/0.168)|0.186 (0.257/0.146)|0.228 (0.314/0.179)|0.224 (0.309/0.176)|0.222 (0.306/0.174)|0.173 (0.238/0.135)|0.225 (0.31/0.176) |
|citeulike180|0.106 (0.145/0.083)|0.097 (0.133/0.077)|0.106 (0.145/0.083)|0.114 (0.156/0.09) |0.102 (0.139/0.08) |0.096 (0.131/0.075)|0.078 (0.107/0.061)|0.058 (0.079/0.045)|0.066 (0.091/0.052)|


Benchmark on [SemEval-2010](https://www.aclweb.org/anthology/S10-1004.pdf) dataset processed by 
[Boudin et al., 16](https://www.aclweb.org/anthology/W16-3917.pdf), where we use Lvl 4 processed dataset.
F1 score, mean precision, and recall (P/R) are reported above.
You can produce those metrics by the following script. LDA are performed on the test corpus. 

```shell script
python ./exmpales/benchmark_semeval.py -m {algrithm-to-test} -e {result-export-directory}
```
