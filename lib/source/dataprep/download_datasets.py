from urllib import request
from urllib.error import URLError
import warnings


def get_files(datasets):
    
    for dataset in datasets:
        BASE_URL = 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall'
        Path = '/workspace/datasets'

        data_url = f'{BASE_URL}/{dataset}_5.json.gz'
        try:
            tmp_file, _ = request.urlretrieve(data_url, f'{Path}/{dataset}_5.json.gz')
        except URLError:
            import ssl
            warnings.warn(
                'Unable to load data securely due to a cetificate problem! '
                'Disabling SSL certificate check.', UserWarning
            )
            # potentially unsafe
            ssl._create_default_https_context = ssl._create_unverified_context
            tmp_file, _ = request.urlretrieve(data_url, f'{Path}/{dataset}_5.json.gz')

if __name__ == "__main__":
    datasets = ['Kidney_Store']
    get_files(datasets)