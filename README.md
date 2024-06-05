# Gromov-delta-estimation
A project provides various functions for calculating Gromov delta-hyperbolicity on a user-item datasets.
Current implementation works with [Amazon](https://jmcauley.ucsd.edu/data/amazon/) and [MovieLens](http://files.grouplens.org/datasets/movielens/) datasets. All calculus are performed with [hypdelta](https://github.com/tnmtvv/hypdelta) module.

## Repo structure
All variations of algos for calculating delta are located in [`lib/source/algo`](./lib/source/algo) folder.<br>
[`notebooks`](./notebooks/) folder contains useful jupiter notebooks for visualizing statistics and charting.<br>[`scripts`](./scripts/) for running current configuration are in scripts folder.

## Positional arguments
```python
Builds csv file with experiment results 


csv_name:             Csv name
config_path:          Path to config file, where dependencies and default parameters are stored
min_batch:            Minimum batch size
min_rank:             Minimum rank
max_batch:            Maximum batch size
max_rank:             Maximum rank 
grid_batch_size:      Num measurements on batch
grid_rank:            Num measurements on rank
-v:                   Flag to print logs
-p:                   Flag to consider min_batch/max_batch as percents

```
Note that all essential directories (for datasets, csvs and svds) along with required dependencies should be set in the config file. See delta_config.yml for an example.

## Example of usage
```python
python scripts/est_delta.py -p -v experiments_delta.csv delta_config.yml 100 3000 100 3000 1 1
```



