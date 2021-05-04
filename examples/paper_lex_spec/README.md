# Reproduce Analysis of LexSpec Paper 
Here we go through our code to reproduce the result described in 
[Back to the Basics: A Quantitative Analysis of Statistical and Graph-Based Term Weighting Schemes for Keyword Extraction](https://arxiv.org/abs/2104.08028).

## Setup
You need [kex library](https://pypi.org/project/kex/).
```shell script
pip install kex
``` 

## Run Experiments
- Run benchmark: MRR/P@k metric for each combination of algorithm and dataset. 
```shell script
python benchmark.py
```

- Get average run time as a proxy for algorithm's complexity measure (note that the run-time depends on environment).
```shell script
python complexity.py
```

- High level statistics for each data.
```shell script
python data_statistics_table.py
```
