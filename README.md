# Gromov-delta-estimation

## Example of usage
```python
Builds csv file with experiment results 

positional arguments:
csv_dir:              Directory where csvs are stored
csv_name:             Csv name
config_path:          Path to config file, where dependencies and default parameters are stored
svd_dir               Directiry where matrices stores
min_batch:            Minimum batch size
min_rank:             Minimum rank
max_batch:            Maximum batch size
max_rank:             Maximum rank 
grid_batch_size:      Num measurments on batch
grid_rank:            Num measurments on rank
-v:                   Flag to print logs
-p:                   Flag to consider min_batch/max_batch as percents

```
Required dependencies can be established in config file as well as default values