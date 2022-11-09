import itertools
import json

import numpy
import os

import requests
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import numpy as np
from app import db
from joblib import dump, load

from app.base.constants.BM_CONSTANTS import df_location
from app.base.db_models import ModelEncodedColumns
from app.base.db_models.ModelFeatures import ModelFeatures
from base.constants.BM_CONSTANTS import api_data_filename, data_files_folder, api_data_folder
from bm.db_helper.AttributesHelper import add_encoded_column_values
from bm.db_helper.DBConnector import DBConnector
from bm.utiles.CVSReader import get_only_file_name, get_file_path
from bm.utiles.utilities import isfloat


class AdjustDataFrame:
    def __init__(self, name):
        self.name = name


pkls_location = 'pkls/'


def encode_data_frame1(data: DataFrame):
    columns_name = data.columns
    encoded_data = data
    data_types = data.dtypes
    for i in range(len(data_types)):
        if data_types[i] != np.int64:
            col_name = columns_name[i]
            oe_style = OneHotEncoder()
            oe_results = oe_style.fit_transform(data[[col_name]])
            pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_).head()
            # encoded_data = encoded_data.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))
            encoded_data = encoded_data.merge(pd.DataFrame(oe_results.toarray()), how='left', left_index=True,
                                              right_index=True)
    return encoded_data


def encode_data_frame(model_id, data: DataFrame, column_type):
    try:
        if column_type != 'F':
            return encode_labels_data_frame(model_id, data)
        else:
            return encode_features_data_frame(model_id, data)
    except  Exception as e:
        print('Ohh -encode_data_frame...Something went wrong.')
        print(e)
        return 0


def encode_features_data_frame(model_id, data: DataFrame, column_type='F'):
    columns_name = data.columns
    encoded_data = []
    data_types = data.dtypes
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
    return data


def encode_labels_data_frame(model_id, data: DataFrame, column_type='L'):
    try:
        columns_name = data.columns
        encoded_data = []
        data_types = data.dtypes
        for i in range(len(data_types)):
            if data_types[i] != np.int64 and data_types[i] != np.float:
                col_name = columns_name[i]
                dummies = pd.get_dummies(data[[col_name]] if data_types[i] != np.float else round(data[[col_name]], 0))
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
        return data
    except  Exception as e:
        print('Ohh -encode_data_frame...Something went wrong.')
        print(e)
        return 0


def encode_data_array(columns_list, data_array):
    data_frame = pd.DataFrame(data_array)
    # Create the mapper
    data_frame_columns = data_frame.columns
    zip_iterator = zip(data_frame_columns, columns_list)
    a_dictionary = dict(zip_iterator)

    data_frame = data_frame.rename(a_dictionary, axis=1)
    data_types = data_frame.dtypes
    columns_name = data_frame.columns
    encoded_data_frame = data_frame
    for i in range(len(data_types)):
        if data_types[i] != np.int64:
            col_name = columns_name[i]
            encoder = ce.OneHotEncoder(cols=col_name, use_cat_names=True)
            encoded_data_frame = encoder.fit_transform(encoded_data_frame)
    print(encoded_data_frame)
    return encoded_data_frame


"""
Function: encode_prediction_data_frame(data: DataFrame)
    Use this function to encode the sent values for prediction from the user.
    The function uses the same encoder that have been used to encode the training and testing data
"""


# def encode_prediction_data_frame(data: DataFrame):
def encode_prediction_data_frame(features_values, column_type):
    # 1- Get the all columns after encoded
    encoded_dataframe_columns = np.asarray(
        ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter_by(
            column_type=column_type).all()).flatten()
    model_features = np.array(ModelFeatures.query.with_entities(ModelFeatures.feature_name).all()).flatten()
    # p_value = ['amil Nadu', '8', '0', '1', '0', '0', '0.5']
    p_value = features_values
    # 2- Match the predicted columns with the encoded columns
    out_put = []
    enc_label = []
    for i in range(len(model_features)):
        get_indexes = lambda encoded_dataframe_columns, xs: [i for (y, i) in zip(xs, range(len(xs))) if
                                                             encoded_dataframe_columns in y]
        occurrences_indexes = get_indexes(model_features[i], encoded_dataframe_columns)
        number_of_occurrences = len(occurrences_indexes)
        # print('model_features[i] = ' + str(model_features[i]))
        # print('number_of_occurrences = ' + str(number_of_occurrences))
        label_len = len(model_features[i])
        if number_of_occurrences == 1:
            if isfloat(p_value[i] or p_value[i].isnumeric()):
                print(p_value[i])
                out_put.append(p_value[i])
            else:
                print(p_value[i])
                out_put.append(1)
        elif number_of_occurrences > 1:
            predicted_value = p_value[i]
            for j in range(len(occurrences_indexes)):
                # print("occurances_indexes[j]=" + str(occurrences_indexes[j]))
                # print("p_value[occurances_indexes[j]]=" + str(p_value[occurrences_indexes[i]]))
                # print("encoded_dataframe_columns[occurrences_indexes[j]]= " + str( encoded_dataframe_columns[occurrences_indexes[j]]))
                if str(predicted_value) in str(encoded_dataframe_columns[occurrences_indexes[j]]):
                    # print("the predicted_value= " + predicted_value)
                    # print("the encoded_dataframe_column= " + str(encoded_dataframe_columns[occurrences_indexes[j]]))
                    # print("j= " + str(j))
                    out_put.append(1)
                else:
                    out_put.append(0)
        else:
            print('0')
    # 3-
    return out_put


def remove_null_values(data: DataFrame):
    # df1 = data.dropna(axis=1, how='all')    # Remove columns that have all null values
    df1 = data.dropna(axis=0, how='any')  # Remove rows that have any null values

    return df1


def create_figure(csv_file_location, x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#E8E5DA')

    x = x_axis
    y = y_axis

    ax.bar(x, y, color="#304C89")

    plt.xticks(rotation=30, size=5)
    plt.ylabel("Expected Clean Sheets", size=5)

    # save the figure
    image_location = get_file_path(csv_file_location) + get_only_file_name(csv_file_location) + '_plot.png'
    plt.savefig(image_location, dpi=300,
                bbox_inches='tight')

    return image_location


def get_encoded_columns(df_head, column_name):
    encoded_columns = []

    for i in range(len(df_head)):
        if df_head[i].find(column_name) >= 0:
            encoded_columns.append(df_head[i])
    print(encoded_columns)
    return encoded_columns


def decode_predicted_values(model_id, p_value, labels, enc_labels):
    try:
        # p_value = [8, 0, 1, 0, 0, 0.5, 1, 0]
        # lables = ['A', 'BB', 'C', 'D']  # DBBBB
        # enc_labels = ['A', 'BB0.5', 'BB12', 'BB115', 'BB0', 'C', 'D12', 'D1150']  # DB
        out_put = []
        # out_put[8, 12, 0.5, 12]
        for i in range(len(labels)):
            get_indexes = lambda enc_labels, xs: [i for (y, i) in zip(xs, range(len(xs))) if enc_labels in y]
            occurances_indexes = get_indexes(labels[i], enc_labels)
            number_of_occurances = len(occurances_indexes)
            print(labels[i] + "= " + str(number_of_occurances))
            label_len = len(labels[i])
            if number_of_occurances == 1:
                out_put.append(str(p_value[occurances_indexes[0]]))
            elif number_of_occurances > 1:
                predicted_value = p_value[occurances_indexes]
                if 1 in predicted_value:  # Check if there is return value in the encoded values, if no return 0
                    for j in range(len(occurances_indexes)):
                        predicted_value = p_value[occurances_indexes[j]]
                        if predicted_value == 1:
                            real_value = enc_labels[occurances_indexes[j]][label_len:]
                            out_put.append(real_value)
                else:
                    out_put.append('Can not be predicted')
            else:
                print('Nothing')
        print(out_put)
        return out_put
    except Exception as e:
        print('Ohh -decode_predicted_values...Something went wrong.')
        print(e)
        return 0


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return (res)


def deletemodelfiles(*argv):
    try:
        for arg in argv:
            files_in_directory = os.listdir(arg)
            filtered_files = [file for file in files_in_directory if not file.endswith(".gitkeep")]
            for f in filtered_files:
                #os.remove(os.path.join(arg, f))
                path_to_file = os.path.join(arg, f)
                os.remove(path_to_file)
        return 1
    except Exception as e:
        print('Ohh -delete_model_files...Something went wrong.')
        print(e)
        return 0


def encode_one_hot(model_id, data_frame, column_types):
    if column_types == 'F':
        encoded_data = encode_one_hot_features(model_id, data_frame)
    else:
        encoded_data = encode_one_hot_labels(model_id, data_frame)
    return encoded_data


def encode_one_hot_features(model_id, data_frame):
    columns_name = data_frame.columns

    for i in range(len(columns_name)):
        model_encoded_column = {'model_id': model_id, 'column_name': columns_name[i],
                                'column_type': 'F'}
    model_encoded = ModelEncodedColumns(**model_encoded_column)
    db.session.add(model_encoded)
    db.session.commit()
    db.session.close()

    ohc = OneHotEncoder()
    encoded_data = ohc.fit_transform(data_frame)
    dump(ohc, open('F_ohencoder.joblib', 'wb'))  # save the model
    return encoded_data


def encode_one_hot_labels(model_id, data_frame):
    columns_name = data_frame.columns

    for i in range(len(columns_name)):
        model_encoded_column = {'model_id': model_id, 'column_name': columns_name[i],
                                'column_type': 'L'}
    model_encoded = ModelEncodedColumns(**model_encoded_column)
    db.session.add(model_encoded)
    db.session.commit()
    db.session.close()

    ohc = OneHotEncoder()
    encoded_data = ohc.fit_transform(data_frame)
    dump(ohc, open('L_ohencoder.joblib', 'wb'))  # save the model
    return encoded_data


def encode_one_hot_input_features(model_id, data_frame):
    ohc = load('F_ohencoder.joblib')
    print(ohc.categories_)
    encoded_data = ohc.fit_transform(data_frame)
    return encoded_data


def reverse_one_hot(encoded_data, df_headers, column_name):
    ohc = load('L_ohencoder.joblib')
    reversed_data = [{} for _ in range(len(encoded_data))]
    all_categories = list(itertools.chain(*ohc.categories_))
    category_names = ['category_{}'.format(i + 1) for i in range(len(ohc.categories_))]
    category_lengths = [len(ohc.categories_[i]) for i in range(len(ohc.categories_))]

    for row_index, feature_index in zip(*encoded_data.nonzero()):
        category_value = all_categories[feature_index]
        category_name = get_category_name(feature_index, category_names, category_lengths, df_headers)
        reversed_data[row_index][category_name] = category_value
        # reversed_data[row_index][column_name] = y[row_index]

    return reversed_data


def get_category_name(index, names, lengths, df_headers):
    counter = 0
    for i in range(len(lengths)):
        counter += lengths[i]
        if index < counter:
            return df_headers[i]
    raise ValueError('The index is higher than the number of categorical values')


def import_mysql_table_csv(host_name, username, password, database_name, table_name):
    cc = DBConnector()

    crsour = cc.create_mysql_connection(host_name, username, password, database_name)
    # s_q_l = "SELECT table_name  FROM INFORMATION_SCHEMA.TABLES WHERE (TABLE_SCHEMA = 'heroku_f01b6802cd615be')"
    table_columns = "SELECT column_name  FROM INFORMATION_SCHEMA.COLUMNS WHERE (TABLE_NAME = '{}')".format(table_name)
    table_data = "SELECT *  FROM {}".format(table_name)
    mycursor = crsour.cursor()
    mycursor.execute(table_columns)
    column_names = numpy.array(mycursor.fetchall()).flatten()
    mycursor.execute(table_data)
    df = pd.DataFrame(mycursor.fetchall(), columns=column_names)
    file_location = df_location + "/{}.csv".format(table_name)
    df.to_csv(file_location, index=False)
    mycursor.close()
    return file_location

def export_mysql_query_to_csv(host_name, username, password, database_name, query_statement):
    db_connector = DBConnector()
    conn = db_connector.create_mysql_connection(host_name, username, password, database_name)

    sql_query = pd.read_sql_query(query_statement, con=conn)  # here, the 'conn' is the variable that contains your database connection information from step 2
    df = pd.DataFrame(sql_query)
    conn.close()

    file_location = df_location + "/{}.csv".format(database_name)
    df.to_csv(file_location, index=False)

    # Remove empty columns
    data = pd.read_csv(file_location)
    data = data.dropna(axis=1, how='all')
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    data.to_csv(file_location, index=False)
    data = pd.read_csv(file_location)

    # Check if the dataset if engough
    count_row = data.shape[0]

    return file_location, count_row

def export_api_respose_to_csv(api_url, request_type, root_node: None, request_parameters:None):
    try:
        api_response = requests.get(url=api_url, json=request_parameters) if request_type == 'type_get' else requests.post(url=api_url, json=request_parameters)

        if (api_response.status_code != 200):
            raise Exception("Error calling the API.")

        # Create json file
        json_response = json.loads(api_response.text)
        df =  pd.json_normalize(json_response) if root_node == None else pd.json_normalize(json_response[root_node])
        df = pd.DataFrame(df)
        data_file_path = api_data_folder + "{}".format(api_data_filename)
        df.to_csv(data_file_path, index=False)
        data = pd.read_csv(data_file_path)

        # Check if the dataset if engough
        count_row = data.shape[0]

        return data_file_path, count_row

    except Exception as e:
        print('Ohh -delete_model_files...Something went wrong.')
        print(e)
        return e

# Delete the orginal data file and create sample data file with the same name
def convert_data_to_sample(ds_file_location, no_of_sample=5):
    df = pd.read_csv(ds_file_location, sep=",")
    df.to_pickle('portfolio.pkl')
    #data_sample = (df.sample(n=no_of_sample, replace=True))
    #os.remove(ds_file_location)  # Delete original data source file
    #data_sample.to_csv(ds_file_location, index=False)
    return 0
