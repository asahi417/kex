# [Kex](https://pypi.org/project/kex/)
*Kex* is a python library for unsurpervised keyword extractions: 
- [Easy interface for keyword extraction with a variety of algorithms](#extract-keywords-with-kex)
- [Quick benchmarking over 15 English public datasets](#benchmark-on-15-public-datasets)
- [Custom keyword extractor implementation support](#implement-custom-extractor-with-kex)

## Get Started
Install via pip
```shell script
pip install kex
```

## Extract Keywords with `kex`
Built-in algorithms in *kex* is below:
- `FirstN`: heuristic baseline to pick up first n phrases as keywords 
- `TF`: scoring by term frequency
- `TFIDF`: scoring by TFIDF
- `LexSpec`: scoring by lexical specificity
- `TextRank`: [Mihalcea et al., 04](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
- `SingleRank`: [Wan et al., 08](https://aclanthology.info/pdf/C/C08/C08-1122.pdf)
- `TopicalPageRank`: [Liu et al.,10](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf)
- `SingleTPR`: [Sterckx et al.,15](https://core.ac.uk/download/pdf/55828317.pdf)
- `TopicRank`: [Bougouin et al.,13](http://www.aclweb.org/anthology/I13-1062)
- `PositionRank`: [Florescu et al.,18](http://people.cs.ksu.edu/~ccaragea/papers/acl17.pdf)
- `TFIDFRank`: SingleRank + TFIDF based population term
- `LexRank`: SingleRank + lexical specificity based population term

Basic usage:

```python
>>> import kex
>>> model = kex.SingleRank()  # any algorithm listed above
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
[{'stemmed': 'non-keyphras word vector',
  'pos': 'ADJ NOUN NOUN',
  'raw': ['non-keyphrase word vectors'],
  'offset': [[47, 49]],
  'count': 1,
  'score': 0.06874471825637762,
  'n_source_tokens': 112},
 {'stemmed': 'semant regular word',
  'pos': 'ADJ NOUN NOUN',
  'raw': ['semantic regularities words'],
  'offset': [[28, 32]],
  'count': 1,
  'score': 0.06001468574146248,
  'n_source_tokens': 112}]
```

### Compute a prior
Algorithms such as `TF`, `TFIDF`, `TFIDFRank`, `LexSpec`, `LexRank`, `TopicalPageRank`, and `SingleTPR` need to compute
a prior distribution beforehand:  

```python
>>> import kex
>>> model = kex.SingleTPR()
>>> test_sentences = ['documentA', 'documentB', 'documentC']
>>> model.train(test_sentences, export_directory='./tmp')
``` 

Priors are cached and can be loaded on the fly:
```python
>>> import kex
>>> model = kex.SingleTPR()
>>> model.load('./tmp')
```

### Supported language
Currently algorithms are available only in English, but soon we will relax the constrain to allow other language to be supported.

## Benchmark on 15 Public Datasets
Users can fetch 15 public keyword extraction datasets via [`kex.get_benchmark_dataset`](./kex/_get_dataset.py#L41).

```python
>>> import kex
>>> json_line, language = kex.get_benchmark_dataset('Inspec', keep_only_valid_label=False)
>>> json_line[0]
{
    'keywords': ['kind infer', 'type check', 'overload', 'nonstrict pure function program languag', ...],
    'source': 'A static semantics for Haskell\nThis paper gives a static semantics for Haskell 98, a non-strict ...',
    'id': '1053.txt'
}
```

High level statistics of each dataset can be found [here](./benchmark/data_statistics.csv), and the benchmark results below:
- [***Precision at top 5***](./benchmark/result.5.precision.fixed.csv)
- [***Precision at top 10***](./benchmark/result.10.precision.fixed.csv) 
- [***MRR***](./benchmark/result.mrr.csv)
- [***Complexity (process time)***](./benchmark/complexity.csv) 
 
A prior distributions are computed within each dataset, and complexity is an average over 100 trial on Inspec dataset.
To reproduce the above benchmark results, please take a look an [example script](./examples/benchmark.py).

## Implement Custom Extractor with `kex`
We provide an API to run a basic pipeline for preprocessing, by which one can implement a custom keyword extractor.

```python
import kex

class CustomExtractor:
    """ Custom keyword extractor example: First N keywords extractor """

    def __init__(self, maximum_word_number: int = 3):
        """ First N keywords extractor """
        self.phrase_constructor = kex.PhraseConstructor(maximum_word_number=maximum_word_number)

    def get_keywords(self, document: str, n_keywords: int = 10):
        """ Get keywords

         Parameter
        ------------------
        document: str
        n_keywords: int

         Return
        ------------------
        a list of dictionary consisting of 'stemmed', 'pos', 'raw', 'offset', 'count'.
        eg) {'stemmed': 'grid comput', 'pos': 'ADJ NOUN', 'raw': ['grid computing'], 'offset': [[11, 12]], 'count': 1}
        """
        phrase_instance, stemmed_tokens = self.phrase_constructor.tokenize_and_stem_and_phrase(document)
        sorted_phrases = sorted(phrase_instance.values(), key=lambda x: x['offset'][0][0])
        return sorted_phrases[:min(len(sorted_phrases), n_keywords)]
```
