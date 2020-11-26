# Grapher

<p align="center">
  <img src="./asset/topic_rank_fig.png" width="800">
  <br><i>Fig 1: TopicRank visualization (Bougouin et al.,13) </i>
</p>


*Grapher* is a python library for unsurpervised keyword extractions: 
- Easy interface for keyword extraction via python
- Quick benchmarking over [16 English public datasets](#benchamrk) for grapher preset methods
- Modules to support implementing [custom keyword extractor](#implement-custom-method-with-grapher)

## Get Started
Install via pip
```shell script
pip install git+https://github.com/asahi417/grapher
```

or clone and install

```shell script
git clone https://github.com/asahi417/grapher
cd grapher
pip install .
```

## Extract Keywords with grapher
*Grapher* retrieves keywords given a document with various algorithms:
- `TFIDF`: a simple statistic baseline
- `TextRank`: [Mihalcea et al., 04](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
- `SingleRank`: [Wan et al., 08](https://aclanthology.info/pdf/C/C08/C08-1122.pdf)
- `TopicRank`: [Bougouin et al.,13](http://www.aclweb.org/anthology/I13-1062)
- `PositionRank`: [Florescu et al.,18](http://people.cs.ksu.edu/~ccaragea/papers/acl17.pdf)
- `MultipartiteRank`: [Boudin 18](https://arxiv.org/pdf/1803.08721.pdf)
- `ExpandRank`: [Wan et al., 08](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf)
- `TopicalPageRank`: [Liu et al.,10](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2010.pdf)
- `SingleTPR`: [Sterckx et al.,15](https://core.ac.uk/download/pdf/55828317.pdf)

Basic usage:

```python
import grapher
model = grapher.SingleRank()  # any algorithm listed above
sample = '''
We propose a novel unsupervised keyphrase extraction approach that filters candidate keywords using outlier detection.
It starts by training word embeddings on the target document to capture semantic regularities among the words. It then
uses the minimum covariance determinant estimator to model the distribution of non-keyphrase word vectors, under the
assumption that these vectors come from the same distribution, indicative of their irrelevance to the semantics
expressed by the dimensions of the learned vector representation. Candidate keyphrases only consist of words that are
detected as outliers of this dominant distribution. Empirical results show that our approach outperforms state
of-the-art and recent unsupervised keyphrase extraction methods.
'''
model.get_keywords(sample, n_keywords=2)

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

### Algorithm with prior
Algorithms with priors (`TFIDF`, `ExpandRank`, `TopicalPageRank`) need to be trained beforehand:
```python
import grapher
model = grapher.SingleTPR()
test_sentences = ['documentA', 'documentB', 'documentC']
model.train(test_sentences, export_directory='./tmp')
``` 

Priors are cached and can be loaded on the fly:
```python
import grapher
model = grapher.SingleTPR()
model.load('./tmp')
```

### Supported Language
Currently, algorithms are available only in English, but soon we will relax the constrain to allow other language to be supported.
The dependency is mainly due to the Part-of-Speech tagger and the word stemmer.

## Implement Custom Method with grapher
Here is a brief example to create a custom extractor with grapher.

```python
import grapher

class CustomExtractor:
    """ Custom keyword extractor example: First N keywords extractor """

    def __init__(self, maximum_word_number: int = 3):
        """ First N keywords extractor """
        self.phrase_constructor = grapher.PhraseConstructor(maximum_word_number=maximum_word_number)

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

## Benchamrk
We enable users to fetch 16 public keyword extraction datasets via 
[`grapher.get_benchmark_dataset`](./grapher/_get_dataset.py) module.
By which, we provide an [example script](./examples/benchmark.py) to run benchmark over preset algorithms.
To benchmark [custom algorithm](#implement-custom-method-with-grapher), see [the other script](./examples/benchmark_custom_model.py).

### Results
All the metrics are in the format of micro F1 score (recall/precision).
Model priors are computed within each dataset.
 
***top 5***  

|              | positionrank       | tfidf               | singletpr          | singlerank          | lexrank            | topicrank          | textrank            | lexspec             | expandrank         | 
|--------------|--------------------|---------------------|--------------------|---------------------|--------------------|--------------------|---------------------|---------------------|--------------------| 
| semeval2017  | 15.1 (34.08/9.7)   | 15.98 (36.06/10.26) | 12.22 (27.59/7.85) | 16.45 (37.12/10.56) | 16.39 (37.0/10.53) | 12.06 (27.22/7.75) | 15.96 (36.02/10.25) | 15.78 (35.62/10.14) | 16.57 (37.4/10.64) | 
| pubmed       | 3.93 (8.6/2.54)    | 2.65 (5.8/1.72)     | 3.89 (8.52/2.52)   | 4.31 (9.44/2.79)    | 3.54 (7.76/2.3)    | 2.28 (5.0/1.48)    | 4.38 (9.6/2.84)     | 2.99 (6.56/1.94)    | 3.41 (7.48/2.21)   | 
| semeval2010  | 10.08 (20.74/6.66) | 5.44 (11.19/3.59)   | 5.6 (11.52/3.7)    | 6.12 (12.59/4.04)   | 6.0 (12.35/3.96)   | 6.08 (12.51/4.02)  | 4.92 (10.12/3.25)   | 5.48 (11.28/3.62)   | 5.52 (11.36/3.65)  | 
| krapivin2009 | 11.66 (12.06/11.3) | 5.81 (6.01/5.63)    | 6.51 (6.73/6.3)    | 7.06 (7.3/6.84)     | 7.57 (7.82/7.33)   | 5.7 (5.89/5.52)    | 4.86 (5.03/4.71)    | 6.46 (6.68/6.25)    | 7.52 (7.77/7.28)   | 
| inspec       | 13.33 (25.48/9.03) | 12.5 (23.89/8.47)   | 12.35 (23.6/8.36)  | 14.95 (28.56/10.12) | 13.72 (26.21/9.29) | 11.59 (22.15/7.85) | 14.8 (28.28/10.02)  | 12.36 (23.62/8.37)  | 13.92 (26.6/9.43)  | 
| nguyen2007   | 11.09 (18.85/7.85) | 7.88 (13.4/5.58)    | 7.26 (12.34/5.14)  | 7.77 (13.21/5.5)    | 8.89 (15.12/6.3)   | 6.58 (11.2/4.66)   | 5.51 (9.38/3.91)    | 8.22 (13.97/5.82)   | 8.67 (14.74/6.14)  | 
| fao30        | 6.45 (24.0/3.72)   | 5.19 (19.33/3.0)    | 8.42 (31.33/4.86)  | 7.88 (29.33/4.55)   | 6.45 (24.0/3.72)   | 4.12 (15.33/2.38)  | 7.52 (28.0/4.34)    | 6.27 (23.33/3.62)   | 7.16 (26.67/4.14)  | 
| kdd          | 6.05 (5.51/6.72)   | 5.73 (5.22/6.36)    | 4.89 (4.45/5.42)   | 5.79 (5.27/6.43)    | 5.73 (5.22/6.36)   | 4.34 (3.95/4.81)   | 5.41 (4.93/6.01)    | 5.47 (4.98/6.07)    | 5.88 (5.35/6.52)   | 
| fao780       | 6.39 (8.29/5.2)    | 6.94 (9.01/5.65)    | 9.4 (12.2/7.64)    | 9.48 (12.3/7.71)    | 9.02 (11.71/7.34)  | 5.04 (6.55/4.1)    | 8.66 (11.25/7.05)   | 7.87 (10.22/6.4)    | 9.02 (11.71/7.34)  | 
| wiki20       | 3.7 (15.0/2.11)    | 4.69 (19.0/2.68)    | 4.2 (17.0/2.39)    | 3.46 (14.0/1.97)    | 4.2 (17.0/2.39)    | 2.96 (12.0/1.69)   | 1.98 (8.0/1.13)     | 3.95 (16.0/2.25)    | 4.2 (17.0/2.39)    | 
| citeulike180 | 8.73 (19.56/5.62)  | 5.8 (13.01/3.73)    | 10.63 (23.83/6.84) | 10.04 (22.51/6.46)  | 9.56 (21.42/6.15)  | 4.49 (10.05/2.89)  | 9.85 (22.08/6.34)   | 7.12 (15.96/4.58)   | 9.36 (20.98/6.02)  | 

***top 10***  

|              | positionrank        | tfidf               | singletpr           | singlerank          | lexrank             | topicrank           | textrank            | lexspec             | expandrank          | 
|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------| 
| semeval2017  | 21.36 (29.45/16.76) | 22.48 (30.99/17.64) | 18.63 (25.68/14.61) | 22.98 (31.68/18.03) | 22.42 (30.91/17.59) | 17.26 (23.79/13.54) | 22.79 (31.42/17.88) | 22.22 (30.63/17.43) | 22.75 (31.36/17.85) | 
| pubmed       | 4.59 (6.18/3.66)    | 2.79 (3.76/2.22)    | 4.37 (5.88/3.48)    | 4.88 (6.56/3.88)    | 3.91 (5.26/3.11)    | 3.23 (4.34/2.57)    | 5.1 (6.86/4.06)     | 3.27 (4.4/2.6)      | 3.84 (5.16/3.05)    | 
| semeval2010  | 11.2 (14.32/9.19)   | 6.4 (8.19/5.26)     | 7.21 (9.22/5.92)    | 8.01 (10.25/6.58)   | 8.05 (10.29/6.61)   | 8.3 (10.62/6.82)    | 5.99 (7.65/4.91)    | 6.69 (8.56/5.5)     | 7.72 (9.88/6.34)    | 
| krapivin2009 | 11.42 (8.76/16.41)  | 5.62 (4.31/8.08)    | 7.29 (5.59/10.47)   | 7.61 (5.83/10.93)   | 7.68 (5.89/11.04)   | 6.46 (4.96/9.29)    | 5.34 (4.1/7.68)     | 6.55 (5.02/9.41)    | 7.65 (5.86/10.99)   | 
| inspec       | 17.88 (21.56/15.28) | 17.09 (20.6/14.6)   | 16.68 (20.11/14.25) | 19.46 (23.46/16.63) | 18.2 (21.94/15.55)  | 15.09 (18.2/12.9)   | 19.35 (23.33/16.53) | 16.89 (20.36/14.43) | 18.44 (22.23/15.75) | 
| nguyen2007   | 12.13 (13.35/11.12) | 8.22 (9.04/7.53)    | 8.48 (9.33/7.77)    | 9.39 (10.33/8.61)   | 10.35 (11.39/9.49)  | 8.26 (9.09/7.57)    | 7.05 (7.75/6.46)    | 9.26 (10.19/8.49)   | 10.13 (11.15/9.29)  | 
| fao30        | 9.0 (19.0/5.89)     | 5.68 (12.0/3.72)    | 10.1 (21.33/6.62)   | 9.79 (20.67/6.41)   | 8.37 (17.67/5.48)   | 4.74 (10.0/3.1)     | 8.68 (18.33/5.69)   | 6.47 (13.67/4.24)   | 8.21 (17.33/5.38)   | 
| kdd          | 4.83 (3.4/8.3)      | 4.71 (3.32/8.1)     | 4.3 (3.03/7.39)     | 4.6 (3.25/7.91)     | 4.75 (3.35/8.17)    | 3.31 (2.33/5.68)    | 4.43 (3.13/7.62)    | 4.47 (3.15/7.68)    | 4.77 (3.36/8.2)     | 
| fao780       | 8.17 (7.34/9.2)     | 6.1 (5.48/6.87)     | 8.91 (8.01/10.04)   | 8.75 (7.87/9.86)    | 8.25 (7.42/9.3)     | 5.41 (4.87/6.1)     | 8.05 (7.24/9.07)    | 7.1 (6.38/8.0)      | 8.27 (7.43/9.32)    | 
| wiki20       | 4.62 (10.5/2.96)    | 5.05 (11.5/3.24)    | 5.05 (11.5/3.24)    | 4.84 (11.0/3.1)     | 5.27 (12.0/3.38)    | 5.05 (11.5/3.24)    | 2.64 (6.0/1.69)     | 4.4 (10.0/2.82)     | 5.71 (13.0/3.66)    | 
| citeulike180 | 10.6 (14.54/8.35)   | 6.62 (9.07/5.21)    | 11.4 (15.63/8.97)   | 10.56 (14.48/8.32)  | 9.57 (13.11/7.53)   | 5.78 (7.92/4.55)    | 10.17 (13.93/8.0)   | 7.77 (10.66/6.12)   | 9.73 (13.33/7.66)   | 


***top 15***  

|              | positionrank        | tfidf               | singletpr           | singlerank          | lexrank             | topicrank           | textrank            | lexspec             | expandrank          | 
|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------| 
| semeval2017  | 23.74 (25.77/22.0)  | 25.18 (27.34/23.34) | 21.62 (23.48/20.04) | 24.97 (27.11/23.14) | 24.86 (26.99/23.04) | 19.7 (21.39/18.26)  | 24.88 (27.02/23.06) | 25.05 (27.19/23.21) | 25.16 (27.32/23.32) | 
| pubmed       | 4.88 (5.19/4.6)     | 2.96 (3.15/2.79)    | 4.41 (4.69/4.16)    | 4.83 (5.13/4.55)    | 3.99 (4.24/3.76)    | 3.64 (3.87/3.43)    | 5.18 (5.51/4.89)    | 3.25 (3.45/3.06)    | 4.01 (4.27/3.79)    | 
| semeval2010  | 11.41 (11.63/11.2)  | 6.68 (6.8/6.55)     | 7.94 (8.09/7.79)    | 8.59 (8.75/8.43)    | 8.43 (8.59/8.27)    | 9.29 (9.47/9.11)    | 6.81 (6.94/6.68)    | 6.94 (7.08/6.82)    | 8.37 (8.53/8.22)    | 
| krapivin2009 | 10.14 (6.88/19.32)  | 5.21 (3.53/9.93)    | 7.04 (4.77/13.41)   | 7.2 (4.88/13.72)    | 7.02 (4.76/13.37)   | 6.29 (4.26/11.98)   | 5.19 (3.52/9.89)    | 5.84 (3.96/11.13)   | 7.02 (4.76/13.38)   | 
| inspec       | 18.48 (17.93/19.06) | 18.17 (17.63/18.74) | 17.78 (17.25/18.34) | 19.3 (18.73/19.91)  | 18.85 (18.29/19.45) | 15.48 (15.02/15.96) | 19.35 (18.78/19.96) | 17.99 (17.45/18.55) | 19.05 (18.48/19.65) | 
| nguyen2007   | 11.27 (10.14/12.67) | 8.08 (7.27/9.09)    | 9.28 (8.36/10.44)   | 9.92 (8.93/11.16)   | 10.17 (9.15/11.44)  | 8.65 (7.78/9.72)    | 7.44 (6.7/8.37)     | 8.58 (7.72/9.65)    | 9.96 (8.96/11.2)    | 
| fao30        | 9.74 (15.33/7.14)   | 5.65 (8.89/4.14)    | 9.88 (15.56/7.24)   | 9.88 (15.56/7.24)   | 9.32 (14.67/6.83)   | 5.79 (9.11/4.24)    | 9.32 (14.67/6.83)   | 6.49 (10.22/4.76)   | 8.89 (14.0/6.51)    | 
| kdd          | 3.8 (2.42/8.85)     | 3.76 (2.39/8.75)    | 3.62 (2.3/8.43)     | 3.7 (2.36/8.62)     | 3.77 (2.4/8.78)     | 2.69 (1.71/6.26)    | 3.67 (2.34/8.56)    | 3.72 (2.37/8.65)    | 3.8 (2.42/8.85)     | 
| fao780       | 7.58 (5.8/10.91)    | 5.53 (4.24/7.96)    | 7.85 (6.02/11.31)   | 7.69 (5.89/11.07)   | 7.25 (5.55/10.44)   | 5.17 (3.96/7.45)    | 7.27 (5.57/10.47)   | 6.06 (4.64/8.72)    | 7.25 (5.55/10.44)   | 
| wiki20       | 4.75 (8.0/3.38)     | 5.15 (8.67/3.66)    | 5.15 (8.67/3.66)    | 5.35 (9.0/3.8)      | 6.53 (11.0/4.65)    | 5.74 (9.67/4.08)    | 3.56 (6.0/2.54)     | 4.55 (7.67/3.24)    | 6.53 (11.0/4.65)    | 
| citeulike180 | 10.72 (11.58/9.98)  | 6.27 (6.78/5.84)    | 10.62 (11.48/9.88)  | 10.28 (11.11/9.57)  | 9.24 (9.98/8.6)     | 5.8 (6.27/5.4)      | 9.95 (10.75/9.26)   | 7.42 (8.01/6.9)     | 9.88 (10.67/9.19)   | 


***computation time (sec)***

|dataset     |positionrank       |tfidf              |singletpr          |singlerank         |lexrank            |topicrank          |textrank           |lexspec            |expandrank         |
|------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|semeval2017 |8.91               |13.97              |17.86              |8.31               |23.6               |11.04              |8.38               |7.53               |23.59              |
|pubmed      |185.26             |289.55             |335.76             |165.76             |323.32             |483.78             |165.19             |152.23             |322.69             |
|semeval2010 |209.72             |296.94             |342.7              |181.62             |518.9              |565.02             |184.42             |157.28             |520.0              |
|krapivin2009|1690.15            |2617.37            |2898.71            |1522.89            |2841.58            |4234.66            |1518.54            |1361.71            |2864.26            |
|inspec      |25.91              |38.98              |48.58              |25.4               |44.63              |27.85              |25.45              |21.21              |45.82              |
|nguyen2007  |98.37              |156.68             |171.84             |88.83              |168.58             |226.01             |88.66              |82.36              |169.25             |
|fao30       |14.11              |21.82              |24.23              |12.52              |23.82              |41.63              |12.45              |11.34              |23.74              |
|kdd         |6.01               |8.84               |10.9               |5.81               |10.34              |6.62               |5.76               |4.89               |10.4               |
|fao780      |366.82             |555.48             |624.91             |324.49             |610.88             |1137.08            |323.73             |292.54             |614.0              |
|wiki20      |12.22              |18.05              |21.69              |10.92              |33.49              |37.04              |10.8               |9.35               |32.64              |
|citeulike180|87.57              |132.18             |149.7              |77.5               |145.95             |306.59             |77.31              |70.14              |146.21             |

