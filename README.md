# ALECE: An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads

We propose an Attention-based LEarned Cardinality Estimator (ALECE~for short) mainly for SPJ queries. The core idea is to discover the implicit relationships between queries and underlying data by using attention mechanisms in ALECE's two modules built on top of carefully designed featurizations for data and queries. In particular, the data-encoder module makes organic aggregations among all attributes in the database and learn their joint distribution information, whereas the query-analyzer module builds a bridge between the query and data featurizations.

## Requirements

```bash
- Python 3.8+
- Tensorflow 2.10, numpy, scipy, psycopg2, argparse
- gcc 9+, openssl, cmake, readline
```

## Dataset

```bash
- cd ALECE && mkdir data
- Download the STATS dataset (https://drive.google.com/file/d/18V9MhlN_5PmFOhKdzUcAIIOXYc1sYtvK/view?usp=sharing)
- Put STATS.tar.gz in data/;
- tar zxvf STATS.tar.gz
```


## First thing to use the code
- You need to make some configurations of the data locations. Run the following scripts.
```bash
- cd ALECE/src
- python init/initialize.py
```



## Benchmark Preparation

#### Install PostgreSQL (in Linux):

```bash
- cd ALECE
- wget https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
- tar xvf postgresql-13.1.tar.bz2 && cd postgresql-13.1
- patch -s -p1 < ../pg_modify.patch
- ./configure --prefix=/usr/local/pgsql/13.1 --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O0"
- make -j 64 && sudo make install
- echo 'export PATH=/usr/local/pgsql/13.1/bin:$PATH' >> ~/.bashrc
- echo 'export LD_LIBRARY_PATH=/usr/local/pgsql/13.1/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
- source ~/.bashrc
```
- You need to specify the directory '$PG_DATADIR$' to put the database data and create a database whose name is the same with your account name of the Linux system. Suppose your account name is 'Tom' and your postgresql data directory is '/home/Tom/pg_data', run the following scripts.
```bash
- initdb -D /home/Tom/pg_data
- set port = 4321 in /home/Tom/pg_data/postgresql.conf
- pg_ctl -D /home/Tom/pg_data start
- psql -p 4321 dbname=postgres
- create database Tom;
```

- Then, '$PG_DATADIR$' in arg_parser/arg_parser.py and '$PG_USER$' in utils/pg_utils.py needs to be replaced with '/home/Tom/pg_data' and 'Tom', respectively.
You can also conduct both replacements by running the following scripts.
```bash
- cd ALECE/src
- python init/initialize.py --PG_DATADIR /home/Tom/pg_data --PG_USER Tom
```

#### How to Generate Sub-Queries and Workload?

```bash
- In data/STATS/workload/static, you will find two sql files: train_queries.sql and test_queries.sql.
- python benchmark/sub_queries_generator.py --data STATS --wl_data_type train (for train_queries.sql) 
- python benchmark/sub_queries_generator.py --data STATS --wl_data_type train (for test_queries.sql)
- We have provided three dynamic workloads: Insert-heavy, Update-heavy and Dist-shift. 
- Each workload is the mix of the training (sub-)queries, testing (sub-)queries and insert/delete/update statements.
- You can also randomly mix the training/testing (sub-)queries with data manipulation statements to build your own dynamic workload.
```

#### Citation
- If you find the code useful, please cite our paper:

```bash
@article{li2023alece,
  author       = {Pengfei Li and
                  Wenqing Wei and
                  Rong Zhu and
                  Bolin Ding and
                  Jingren Zhou and
                  Hua Lu},
  title        = {{ALECE}: An Attention-based Learned Cardinality Estimator for {SPJ}
                  Queries on Dynamic Workloads},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {17},
  number    = {2},
  pages     = {197--210},
  year      = {2023}
}


```

## ALECE Training
- Suppose we hope to use the training part of the Insert-heavy workload to train ALECE and make estimations for the testing sub-queries in the evaluation part of the same workload, run the follwing scripts. Note that the first execution will take several more minutes because the histograms and features of the queries in the whole workload need to be built.
```bash
- cd ALECE/src;
- python train.py --model ALECE --data STATS --wl_type ins_heavy
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Then, in the directory `ALECE/exp/STATS/e2e`, you will find a file `ALECE_STATS_ins_ins.txt` which covers the estimated cardinalities of the testing sub-queries in the Insert-heavy workload.

- Our code supports to use the training part of workload A to train ALECE but make estimations for the testing sub-queries in the evaluation part of workload B. Suppose A and B are Insert-heavy and Dist-shift, respectively, run the following scripts.

```bash
- cd ALECE/src;
- python train.py --model ALECE --data STATS --wl_type ins_heavy --test_wl_type dist_shit
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Then, in the directory `ALECE/exp/STATS/e2e`, you will find a file `ALECE_STATS_ins_dist.txt` which covers the estimated cardinalities of the testing sub-queries in the Dist-shift workload.
It is noteworthy that the training dataset comes from another workload.

# End-to-end Evaluation with Our Benchmark
- The results in `ALECE_STATS_ins_ins.txt` can be used to do E2E evaluations for the testing queries in the Insert-heavy workload. First, you need to copy this file into the directory $PG_DATADIR$. Then, run the following scripts.

```bash
- cd ALECE/src;
- python benchmark/e2e_eval.py --model ALECE --data STATS --wl_type ins_heavy
```

- If you want to know the original E2E time of the PostgreSQL with the built-in method on this workload, run the following scripts.
```bash
- cd ALECE/src;
- python benchmark/e2e_eval.py --model pg --data STATS --wl_type ins_heavy
```
