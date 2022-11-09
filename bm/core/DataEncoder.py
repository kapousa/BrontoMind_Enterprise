#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm.contrib import itertools
from joblib import dump, load
import pandas as pd

con = pd.Series(list('abcba'))
print(pd.get_dummies(con))


class DataEncoder:

    def encode_one_hot(self, data_frame):
        columns_name = data_frame.columns
        encoded_data = []
        data_types = data_frame.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                col_name = columns_name[i]
                dummies = pd.get_dummies(data[[col_name]])
                dummies_columns = dummies.columns
                # encoded_data = encoded_data.append(dummies)
                data = data.drop([col_name], axis=1)
                data = pd.concat([data, dummies], axis=1)
                endoced_column = get_encoded_columns(data.columns, col_name)
                # encoder = ce.OneHotEncoder(cols=col_name, use_cat_names=True)
                # data = encoder.fit_transform(data)
                addencodedcolumnvalues = add_encoded_column_values(model_id, col_name, dummies, column_type)
                # encoded_data = encoder.inverse_transform(encoded_data)
            else:
                # encoded_data = encoded_data.append(data[columns_name[i]])
                column_data = data[columns_name[i]]
                data = data.drop(columns_name[i], axis=1)
                data = pd.concat([data, column_data], axis=1)
                model_encoded_column = {'model_id': model_id, 'column_name': columns_name[i],
                                        'column_type': column_type}
                model_encoded = ModelEncodedColumns(**model_encoded_column)
                db.session.add(model_encoded)
                db.session.commit()
                db.session.close()

        ohc = OneHotEncoder()
        encoded_data = ohc.fit_transform(data_frame)
        dump(ohc, 'ohencoder.joblib')  # save the model

        return encoded_data

    def reverse_one_hot(self, X, y, encoder, df_headers, column_name):
        reversed_data = [{} for _ in range(len(y))]
        all_categories = list(itertools.chain(*encoder.categories_))
        category_names = ['category_{}'.format(i + 1) for i in range(len(encoder.categories_))]
        category_lengths = [len(encoder.categories_[i]) for i in range(len(encoder.categories_))]

        for row_index, feature_index in zip(*X.nonzero()):
            category_value = all_categories[feature_index]
            category_name = self.get_category_name(feature_index, category_names, category_lengths, df_headers)
            reversed_data[row_index][category_name] = category_value
            reversed_data[row_index][column_name] = y[row_index]

        from joblib import dump, load
        dump(ohc, 'filename.joblib')  # save the model
        ohc1 = load('filename.joblib')  # load and reuse the model
        return reversed_data

    def get_category_name(index, names, lengths, df_headers):
        counter = 0
        for i in range(len(lengths)):
            counter += lengths[i]
            if index < counter:
                return df_headers[i]
        raise ValueError('The index is higher than the number of categorical values')
