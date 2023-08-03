from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from EstDelta import get_movielens_data, matrix_from_observations, get_reviews
from scipy.sparse.linalg import aslinearoperator


def main():
    DATASETS = [
        # 'AMAZON_FASHION',
        # 'All_Beauty',
        # 'Appliances',
        # 'Arts_Crafts_and_Sewing',
        # 'Automotive',
        # 'Books',
        # 'CDs_and_Vinyl',
        # 'Cell_Phones_and_Accessories',
        # 'Clothing_Shoes_and_Jewelry',
        # 'Digital_Music',
        # 'Electronics',
        # 'Gift_Cards',
        # 'Grocery_and_Gourmet_Food',
        # 'Home_and_Kitchen',
        # 'Industrial_and_Scientific',
        # 'Kindle_Store',
        # 'Luxury_Beauty',
        # 'Magazine_Subscriptions',
        # 'Movies_and_TV',
        # 'Musical_Instruments',
        # 'Office_Products',
        # 'Patio_Lawn_and_Garden',
        # 'Pet_Supplies',
        # 'Prime_Pantry',
        # 'Software',
        # 'Sports_and_Outdoors',
        # 'Tools_and_Home_Improvement',
        # 'Toys_and_Games',
        # 'Video_Games',
        'Movielens_1m'
    ]
    # print('start')
    # get_files(DATASETS)
    # print('done')
    # path = 'E:\datasets'
    path ='C:\work\GitHub\DeltaEstimation\datasets'
    datafiles = [f for f in listdir(path) if isfile(join(path, f))]

    datafiles.sort()
    DATASETS.sort()

    data_dict = {
        'Users':[],
        'Items': [],
        'Interactions': [],
        'Dataset': []
    }

    for i, datafile in enumerate(datafiles):
        print(DATASETS[i])
        if DATASETS[i] == 'Movielens_1m':
            cur_df = get_movielens_data(join(path, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df, dtype=float, itemid='movieid')
        else:
            cur_df = get_reviews(join(path, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df, dtype=float)
        data_dict['Users'].append(len(u_id))
        data_dict['Items'].append(len(i_id))
        data_dict['Interactions'].append(matr_from_observ.getnnz())
        data_dict['Dataset'].append(DATASETS[i])
    df = pd.DataFrame(data_dict)
    df.set_index('Dataset')
    df.to_csv('C:\work\GitHub\DeltaEstimation\datasets_info_ml.csv', index=False)


if __name__ == '__main__':
    main()
