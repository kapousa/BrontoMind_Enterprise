import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import itertools

def reverse_one_hot(X, y, encoder):
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


def get_category_name(index, names, lengths):
    counter = 0
    for i in range(len(lengths)):
        counter += lengths[i]
        if index < counter:
            return names[i]
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

data_frame = pd.DataFrame(data)
data_frame = data_frame[['user_id', 'item_id', 'rating']]

# Delete the dependent variable
ratings = data_frame['rating']
data_frame.drop(columns=['rating'], inplace=True)

ohc = OneHotEncoder()
encoded_data = ohc.fit_transform(data_frame)
print(encoded_data)


new_data = [{'user_id': 'Hatem', 'item_id': 'The BOOLO', 'rating': 18}]
new_dataf = pd.DataFrame(new_data)
new_dataf = new_dataf[['user_id', 'item_id', 'rating']]
ratings = new_dataf['rating']
new_dataf.drop(columns=['rating'], inplace=True)

encoded_data = ohc.fit_transform(new_dataf)

reverse_data = reverse_one_hot(encoded_data, ratings, ohc)
print(pd.DataFrame(reverse_data))