# Gromov-delta-estimation
A project provides various funtions for calculating Gromov delta-hyperbolicity on a user-iterm datasets.
Current implementation works with [Amazon](https://jmcauley.ucsd.edu/data/amazon/) and [MoovieLens](http://files.grouplens.org/datasets/movielens/) datasets.

## Positional arguments
```python
Builds csv file with experiment results 


csv_name:             Csv name
config_path:          Path to config file, where dependencies and default parameters are stored
min_batch:            Minimum batch size
min_rank:             Minimum rank
max_batch:            Maximum batch size
max_rank:             Maximum rank 
grid_batch_size:      Num measurments on batch
grid_rank:            Num measurments on rank
-v:                   Flag to print logs
-p:                   Flag to consider min_batch/max_batch as percents
-c:                   Flag to note the mode, set -c for comparison of realisations

```
Note that all essential directories (for datasets, csvs and svds) along with required dependencies should be set in the config file. See delta_config.yml for an example.


