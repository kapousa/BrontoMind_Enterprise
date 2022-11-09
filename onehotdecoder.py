#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import itertools

import pandas
from sklearn.preprocessing import OneHotEncoder


def reverse_one_hot_dis(X, y, encoder):
    reversed_data = [{} for _ in range(len(y))]
    all_categories = list(itertools.chain(*encoder.categories_))
    category_names = ['category_{}'.format(i + 1) for i in range(len(encoder.categories_))]
    category_lengths = [len(encoder.categories_[i]) for i in range(len(encoder.categories_))]

    for row_index, feature_index in zip(*X.nonzero()):
        category_value = all_categories[feature_index]
        category_name = get_category_name(feature_index, category_names, category_lengths)
        reversed_data[row_index][category_name] = category_value
        reversed_data[row_index]['target'] = y[row_index]

    return reversed_data

def reverse_one_hot(X, y, encoder, df_headers, column_name):
    reversed_data = [{} for _ in range(len(y))]
    all_categories = list(itertools.chain(*encoder.categories_))
    category_names = ['category_{}'.format(i + 1) for i in range(len(encoder.categories_))]
    category_lengths = [len(encoder.categories_[i]) for i in range(len(encoder.categories_))]

    for row_index, feature_index in zip(*X.nonzero()):
        category_value = all_categories[feature_index]
        category_name = get_category_name(feature_index, category_names, category_lengths, df_headers)
        reversed_data[row_index][category_name] = category_value
        reversed_data[row_index][column_name] = y[row_index]

    return reversed_data

def get_category_name(index, names, lengths, df_headers):
    counter = 0
    for i in range(len(lengths)):
        counter += lengths[i]
        if index < counter:
            return df_headers[i]
    raise ValueError('The index is higher than the number of categorical values')

data = [
    {'user_id': 'John', 'item_id': 'The Matrix', 'rating': 5},
    {'user_id': 'John', 'item_id': 'Titanic', 'rating': 1},
    {'user_id': 'John', 'item_id': 'Forrest Gump', 'rating': 2},
    {'user_id': 'John', 'item_id': 'Wall-E', 'rating': 2},
    {'user_id': 'Lucy', 'item_id': 'The Matrix', 'rating': 5},
    {'user_id': 'Lucy', 'item_id': 'Titanic', 'rating': 1},
    {'user_id': 'Lucy', 'item_id': 'Die Hard', 'rating': 5},
    {'user_id': 'Lucy', 'item_id': 'Forrest Gump', 'rating': 2},
    {'user_id': 'Lucy', 'item_id': 'Wall-E', 'rating': 2},
    {'user_id': 'Eric', 'item_id': 'The Matrix', 'rating': 2},
    {'user_id': 'Eric', 'item_id': 'Die Hard', 'rating': 3},
    {'user_id': 'Eric', 'item_id': 'Forrest Gump', 'rating': 5},
    {'user_id': 'Eric', 'item_id': 'Wall-E', 'rating': 4},
    {'user_id': 'Diane', 'item_id': 'The Matrix', 'rating': 4},
    {'user_id': 'Diane', 'item_id': 'Titanic', 'rating': 3},
    {'user_id': 'Diane', 'item_id': 'Die Hard', 'rating': 5},
    {'user_id': 'Diane', 'item_id': 'Forrest Gump', 'rating': 3},
]

data_frame = pandas.DataFrame(data)
data_frame = data_frame[['user_id', 'item_id', 'rating']]
original_data_frame_columns = data_frame.columns.values
ratings = data_frame['rating']
data_frame.drop(columns=['rating'], inplace=True)
df_headers = data_frame.columns.values

ohc = OneHotEncoder()
encoded_data = ohc.fit_transform(data_frame)

print(encoded_data)

from joblib import dump, load
dump(ohc, 'filename.joblib') # save the model
ohc1 = load('filename.joblib') # load and reuse the model

reverse_data = reverse_one_hot(encoded_data, ratings, ohc1, df_headers, 'rating')
rdf = pandas.DataFrame(reverse_data)
print(rdf.reindex(columns=original_data_frame_columns))
