# Quality check
Here is a very rough note, which explains how we can conduct keyphrases quality check. 

## Get keyphrase list
Let's say you have couple of csv files, which contains list of document, under a directory `./sample_data/`.
Firstly, you have to aggregate keyphrases for each single document by following script

***(1) Parse csv files*** 

```
python quality_check/get_csv_keyphrase.py -a TopicRank
```

***(2) Aggregate results***

```
python quality_check/get_aggregate_result.py -a TopicRank
```
 
Then you will have `./quality_check/statistics/TopicRank/statistics.json`, which contains the extracted 
keyphrases for each document.

***(3) Check stopword detection***

Looking over the keyphrases might give you insight to improve stopwords detection, and once you modify
the stopwords detection, you can check if the target junk keyphrases are eliminated by the fixed 
stopwords detection or not by following script.

```
python quality_check/stopword_test.py -a TopicRank
```


## Check the quality of stopword detection
Stopword detection is one of the most important component in keyprhase extraction, so it's necessary 
to validate the performance of it. To do that, you can produce keyphrases without stopwords detection 
 by just proceeding step (1) to (3) with following environment variable.

```
export DEV_MODE=without_stopword
```

Then you will get `./quality_check/statistics/TopicRank/statistics_no_stopword.json`, which is the list of 
keyphrase, without stopwords detection. Now, you will be able to see the eliminated phrases by step (3). 

