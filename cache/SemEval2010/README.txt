[text files: test/train/trial data set]

- regular expression describing the filenames: (C|H|I|J)-\d+.txt.final

--------------------------------------------------------------------------
[statistics of the text files]

- test data set:  100           = C(25), H(25), I(25), J(25)
- train data set: 144           = C(34), H(39), I(35), J(36)
- trial data set:  40           = C(10), H(10), I(10), J(10)
- In total:       244 documents = C(59), H(64), I(60), J(61)

Speciftic statistics of the train files.
* 'of' means the preposition 'of' is present in the keyphrase.
* ' means the appostrophe character is present in the keyphrase.
# of author-assigned keyphrases(train.author) =  563 (of=5,  '=1)
# of reader-assigned keyphrases(train.reader) = 1837 (of=50, '=2)
# of combined keyphrases(train.combined)      = 2239 (of=52, '=3)

Specific statistics of the test files.
# of author-assigned keyphrases(test.author)  =  388 (of=6,  '=0)
# of reader-assigned keyphrases(test.reader)  = 1217 (of=55, '=4)
# of combined keyphrases(test.combined)       = 1482 (of=61, '=4)
# of author-assigned keyphrases not found in the context = 61 (of=2, '=0)

--------------------------------------------------------------------------
[format of text files]

- Lines are wrapped by line length.
- Some words may be missing a hypen('-'), due to the merging preprocessing 
of the dataset that restores full words that were originally broken across 
two lines.
- Section headers start with numbers; some are all in uppercase 
(e.g., 1. INTRODUCTION, 1. Introduction)
- Papers contain "Categories and Subject Descriptors" and "General Terms", 
with only few exceptions.

--------------------------------------------------------------------------
[answer data files]

- We provide both stemmed and lemmatized keyphrases in the train data. 
- To stem keyphrases, we use English Porter stemmer consistently across train 
and test datasets.  Implementations in various computer programming languages 
are available at http://tartarus.org/~martin/PorterStemmer/
- Author-assigned keyphrasesd are already present in the text files. (However, we removed them from the original articles for the task)
- We list the files that contain the answer data files below:

* test/train/trial.author.final : lemmatized author-assigned keyprhases
* test/train/trial.author.stem.final : stemmed author-assigned keyprhases
* test/train/trial.reader.final : lemmatized reader-assigned keyprhases
* test/train/trial.reader.stem.final : stemmed reader-assigned keyprhases
* test/train/trial.combined.final : lemmatized combined keyphrases
* test/train/trial.combined.stem.final : stemmed combined keyphrases

--------------------------------------------------------------------------
[Format of your answer file]

To submit answer files, pleaes ensure that you follow the below conventions, 
and that your files are in UNIX format.  If you Follow the instructions carefully 
(assigning the properly filenames), we will be able to run the evaluation 
script without problems.  Thanks for your cooperation!

- "FILENAME\s:\sKEYPHRASE_LIST" separately by a comma where "\s" means a space.
- Please list 15 candidate keyphrases per document.  Your submission's score will 
be based only on the first 15 answers.

- We accept one type of keyphrase format.
  - stemmed words using "Porter stemmer"

e.g. with document C-1
  stemmed words    = C-1 : keyphras,extract,competit,test,perform

- You should submit three different types of FILENAMES, according to the three 
different types of answer formats. (e.g. "teamname.stm")

- TEAMNAME should be up to 8 alphabetic (plus "hypen") characters in uppercase.  
No digits and other punctuation marks should be present in the team name.

(E.g., If the institution submitting an entry is the University of Melbourne and 
you have two different teams, you can assign team names as "UMELB-A" and "UMELB-B") (e.g. UMELB-B.stm)

--------------------------------------------------------------------------
[performance]
- script name: performance.pl
- usage: perl performance.pl <your_file_name>
  (e.g. "perl performance.pl UMELB-A.stm")

- The output of the script shows the number of documents you answered with the 
micro-average precision, recall and F-score.  The script compares the stemmed keyphrases between your answer set and the provided answer set. 

--------------------------------------------------------------------------
[baselines]
- We compute baselines using TF*IDF n-gram based supervised and unsupervised 
learning systems.
- We use 1, 2, 3-grams as keyphrase candidates.
- We use Maximum Entropy and Naive Bayes to train two supervised baselines.
- In total, there are thus three baselines: 2 using supervised learning and 1 
using unsupervised learning.
- The baseline performance are computed using "performance.pl".

