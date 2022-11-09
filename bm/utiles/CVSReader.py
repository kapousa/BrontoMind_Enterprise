import os

import numpy as np
import pandas as pd
import numpy

from app.base.constants.BM_CONSTANTS import df_location
from bm.utiles.ProjConstants import ProjConstants


class CVSReader:

    def __init__(self, name):
        self.name = name


def getcvsheader(csv_path):
    # open file in read mode
    data = pd.read_csv(csv_path, nrows=0)
    columns_array = numpy.array(data.columns)
    #new_columns_array = [x.replace('/','-') if x.find('/')!=0 else x for x in columns_array]
    return columns_array


def get_cvs_number_of_columns(csv_path):
    # open file in read mode
    data = pd.read_csv(csv_path, nrows=0)
    columns_array = numpy.array(data.columns)
    return len(columns_array)


def reorder_csv_file(file_location, new_coulmns_order):
    # Example: How to call the method: x = reorder_csv_file('C:/Users/hgadallah/Desktop/file.csv',
    # 'C:/Users/hgadallah/Desktop/reordered.csv', ['A', 'B', 'D', 'C'])
    df = pd.read_csv(file_location)
    df_reorder = df[new_coulmns_order]  # rearrange column here
    file_name = get_file_name_with_ext(file_location)
    new_file_name = get_file_path(file_location) + 'temp_' + file_name
    df_reorder.to_csv(new_file_name, index=False)
    os.remove(file_location)
    os.rename(new_file_name, file_location)
    return df


def get_new_headers_list(original_header, predicted_values):
    df = np.array(original_header)
    number_of_predictions = len(predicted_values)
    for i in range(number_of_predictions):
        indices = np.where(df == predicted_values[i])
        df = np.delete(df, indices)
    df = np.append(df, predicted_values)
    return df


def improve_data_file(fname, path, predicted_values) -> object:
    temp_file_path = os.path.join(path, 'temp.csv')
    file_path = os.path.join(path, fname)

    csv_reader = getcvsheader(file_path)
    predictionvalues = numpy.array((predicted_values))
    new_csv_header = get_new_headers_list(csv_reader, predictionvalues)
    reorder_header = reorder_csv_file(file_path, new_csv_header)
    return 1

def get_only_file_name(full_path):
    full_path = full_path
    file_name = full_path.rsplit('/', 1)[1] if full_path.find('/')!= -1 else full_path
    only_file_name = file_name.rsplit('.',1)[0]
    return only_file_name

def get_file_name_with_ext(full_path):
    full_path = full_path
    file_name = full_path.rsplit('/', 1)[1] if full_path.find('/')!= -1 else full_path
    return file_name

def get_file_path(full_path):
    full_path = full_path
    file_path = full_path.rsplit('/', 1)[0] if full_path.find('/')!= -1 else full_path
    return file_path + '/'

def adjust_csv_file(fname, classification_features, classification_label):
    file_path = "%s%s" % (df_location, fname)
    df = pd.read_csv(file_path)
    file_headers = numpy.array(df.columns)
    classification_features = numpy.array(classification_features)
    used_columns = np.append(classification_features, [classification_label])

    for file_header in file_headers:
        if (file_header not in used_columns):
            df.drop(file_header, axis=1, inplace=True)
            print(df.head())

    new_file_name = "%s%s%s" % (file_path, 'temp_', fname)
    df.to_csv(new_file_name, index=False)
    os.remove(file_path)
    os.rename(new_file_name, file_path)

    return 'Success'


