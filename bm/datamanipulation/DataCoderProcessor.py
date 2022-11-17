#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import csv
import datetime

import numpy
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import and_
import datetime as dt

from app import db
from sklearn import preprocessing

from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from bm.db_helper.AttributesHelper import getmodelencodedcolumns, get_model_name
from bm.utiles.CVSReader import getcvsheader
from bm.utiles.Helper import Helper


class DataCoderProcessor:
    pkls_location = 'pkls/'
    category_location = 'app/data/'
    flag = ''

    def __init__(self):
        self.flag = '_'

    def encode_features(self, model_name, model_id, data: DataFrame, column_type='F'):
        columns_name = data.columns
        encoded_columns = []
        data_types = data.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                is_date = Helper.validate_timestring(data.iloc[0, data.columns.get_loc(columns_name[i])])
                data_item = {'model_id': model_id, 'column_name': columns_name[i],
                             'column_type': column_type, 'is_date': is_date}
                encoded_columns.append(data_item)
                col_name = columns_name[i]
                dummies = self.encode_column(col_name, data[[col_name]]) if (
                            is_date == 0) else self.endcode_datetime_column(col_name, data[
                    [col_name]])  # to be added to ent version
                dummies = pd.DataFrame(dummies)
                data = data.drop([col_name], axis=1)
                data.insert(i, col_name, dummies)

        db.session.bulk_insert_mappings(ModelEncodedColumns, encoded_columns)
        db.session.commit()
        db.session.close()

        return data

    def encode_labels(self, model_name, model_id, data: DataFrame):
        column_type = 'L'
        return self.encode_features(model_name, model_id, data, 'L')

    def vectrise_feature_text(self, model_id, data: DataFrame):
        columns_name = data.columns
        encoded_columns = []
        data_types = data.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                data_item = {'model_id': model_id, 'column_name': columns_name[i],
                             'column_type': 'F'}
                encoded_columns.append(data_item)
                col_name = columns_name[i]
                dummies = self.vectorize_column(model_id, col_name, data[[col_name]])
                dummies = pd.DataFrame(dummies)
                data = data.drop([col_name], axis=1)
                data.insert(i, col_name, dummies[1:])

        db.session.bulk_insert_mappings(ModelEncodedColumns, encoded_columns)
        db.session.commit()
        db.session.close()

        return data

    def encode_input_values(self, model_id, features_list, input_values):
        encoded_columns = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                and_(ModelEncodedColumns.model_id == str(model_id),
                     ModelEncodedColumns.column_type == 'F')).all())
        model_encoded_columns = model_encoded_columns.flatten()
        model_name = get_model_name(str(model_id))
        for i in range(len(input_values)):
            input_value = input_values[i].strip()
            if (not input_value.isdigit()) and (features_list[i] in model_encoded_columns):
                col_name = features_list[i]
                pkl_file_location = self.pkls_location + model_name + '/' + col_name + '_pkle.pkl'
                encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                column_data_arr = numpy.array(input_value)
                encoded_values = encoder_pkl.transform(column_data_arr.reshape(-1, 1)) if (
                            Helper.is_time(col_name) == 0) else self.endcode_datetime_column(col_name,
                                                                                             column_data_arr.flatten())
                encoded_columns.append(encoded_values[0])
            else:
                encoded_columns.append(input_value)

        return [encoded_columns]

    def decode_output_values(self, model_id, labels_list, input_values):
        decoded_results = []
        decoded_row = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                and_(ModelEncodedColumns.model_id == str(model_id),
                     ModelEncodedColumns.column_type == 'L')).all())
        model_encoded_columns = model_encoded_columns.flatten()
        model_name = get_model_name(str(model_id))
        for i in range(len(input_values)):
            input_values_row = input_values[i]
            for j in range(len(input_values[i])):
                if labels_list[j] in model_encoded_columns:
                    col_name = labels_list[j]
                    pkl_file_location = self.pkls_location + model_name + '/' + col_name + '_pkle.pkl'
                    encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                    column_data_arr = numpy.array(input_values_row[j])
                    original_value = encoder_pkl.inverse_transform(column_data_arr.reshape(-1, 1)) if (
                                Helper.is_time(col_name) == 0) else self.decode_datetime_value(
                        column_data_arr)
                    decoded_row.append(original_value[0].strip())
                else:
                    decoded_row.append(str(input_values_row[j]))
            decoded_results.append(decoded_row)
        return np.array(decoded_results)

    def decode_category_name(self, model_id, category_column, input_values):
        pkl_file_location = "%%s%s%s%s" % (
        self.pkls_location, get_model_name(model_id), '/', category_column, '_pkle.pkl')
        encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
        original_value = encoder_pkl.inverse_transform(input_values)
        return original_value

    def encode_column(self, model_name, column_name, column_data):
        column_data_arr = column_data.to_numpy()
        column_data_arr = column_data_arr.flatten()
        categories = numpy.unique(column_data_arr)
        labelEnc = preprocessing.LabelEncoder()
        encoded_values = labelEnc.fit_transform(column_data_arr.reshape(-1, 1))
        pkl_file_location = "%s%s%s%s%s" % (self.pkls_location, model_name, '/', column_name, '_pkle.pkl')
        pickle.dump(labelEnc, open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = "%s%s%s%s%s" % (self.category_location, model_name, '/', column_name, '_csv.csv')
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(categories)

        return encoded_values

    def vectorize_column(self, model_id, column_name, column_data):
        column_data_arr = column_data.to_numpy()
        column_data_arr = column_data_arr.flatten()
        categories = numpy.unique(column_data_arr)
        vectorizer = TfidfVectorizer()
        column_data_arr = column_data_arr.flatten()
        column_data_list = list(column_data_arr)
        vectors = vectorizer.fit_transform(column_data_list)
        pkl_file_location = "%s%s%s%s%s" % (self.pkls_location, get_model_name(model_id), '/', column_name, '_pkle.pkl')
        pickle.dump(vectorizer, open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = "%s%s%s%s%s" % (
        self.category_location, get_model_name(model_id), '/', column_name, '_csv.csv')
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(categories)

        return vectors.indptr

    def get_all_categories_values(self, model_id):
        encoded_columns = getmodelencodedcolumns(model_id, 'F')
        all_gategories_values = {}

        if len(encoded_columns) == 0:
            return all_gategories_values

        for i in range(len(encoded_columns)):
            category_file_location = "%s%s%s%s%s" % (
            DataCoderProcessor.category_location, get_model_name(model_id), '/', encoded_columns[i], '_csv.csv')
            category_values = getcvsheader(category_file_location)
            all_gategories_values[encoded_columns[i]] = category_values
        return all_gategories_values

    def endcode_datetime_column(self, col_name, df_col):  # to be added to ent version
        df = df_col
        # df[col_name] = df[col_name].astype(str)
        df[col_name] = pd.to_timedelta(df[col_name])
        df[col_name] = df[col_name].dt.total_seconds().astype(int)

        pkl_file_location = self.pkls_location + col_name + '_pkle.pkl'
        pickle.dump('endcode_datetime_column', open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = self.category_location + col_name + '_csv.csv'
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow('')

        return df.loc[:, col_name]

    def decode_datetime_value(self, number_of_seconds_arr):  # to be added to ent version
        number_of_seconds_arr = number_of_seconds_arr.flatten()
        xx = float(number_of_seconds_arr[0])
        td = [str(datetime.timedelta(seconds=xx))]
        return td

    def endcode_datetime_column_(self, col_name, df_col):  # to be added to ent version
        epoch_time = dt.datetime(2000, 1, 1, 0, 0, 0)
        df = {col_name: df_col}
        df = pd.DataFrame(df)

        df[col_name] = pd.to_datetime(df[col_name], format='%d/%m/%Y %I:%M:%S %p')  # Convert data to datetime object

        df[col_name] = (df[col_name] - epoch_time)
        df[col_name] = df[col_name].dt.total_seconds().astype(int)

        pkl_file_location = self.pkls_location + col_name + '_pkle.pkl'
        pickle.dump('endcode_datetime_column', open(pkl_file_location, 'wb'))

        # save categories
        category_file_location = self.category_location + col_name + '_csv.csv'
        with open(category_file_location, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow('')

        return df.loc[:, col_name]

    def decode_datetime_value_(self, number_of_seconds_arr):  # to be added to ent version
        # epoch time
        epoch_time = dt.datetime(2000, 1, 1, 0, 0, 0)
        df = {'number_of_seconds': number_of_seconds_arr}
        df['number_of_seconds'] = df['number_of_seconds'].astype('timedelta64[s]') + epoch_time
        df['number_of_seconds'] = pd.to_datetime(df['number_of_seconds'], unit='s').dt.time

        return df.loc[:, 'number_of_seconds']
