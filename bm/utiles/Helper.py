#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import datetime
import ftplib
import itertools
import os
import random
from calendar import month_name
from datetime import date

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from werkzeug.utils import secure_filename

from app import config_parser
from app.base.constants.BM_CONSTANTS import html_plots_location, html_short_path, data_files_folder, \
    physical_allowed_extensions, results_path, app_results_path
from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from base.db_models.ModelLookupTables import ModelLookupTable


class Helper:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def generate_unique_random_nlist(self, start_index, end_index, length):
        # Generate n unique random numbers within a range
        num_list = random.sample(range(start_index, end_index), length)
        print(num_list)

        return numpy.array(num_list)

    def previous_n_months(n):
        current_month_idx = date.today().month - 1  # Value is now (0-11)
        months_list = []
        for i in range(1, n + 1):
            # The mod operator will wrap the negative index back to the positive one
            previous_month_idx = (current_month_idx - i) % 12  # (0-11 scale)
            m = int(previous_month_idx + 1)
            months_list.append(month_name[m])
        return np.flip(months_list)

    @staticmethod
    def plot_confusion_matrix_as_image(cm, file_name, target_names, title='Confusion matrix', cmap=None,
                                       normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        html_file_location = html_plots_location + file_name + ".html"
        html_path = html_short_path + file_name + ".html"
        #   plotly.offline.plot(plt, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, labels, title):
        # cm : confusion matrix list(list)
        # labels : name of the data list(str)
        # title : title for the heatmap
        cm_sum = sum(sum(cm))
        data = go.Heatmap(z=cm, y=labels, x=labels)
        annotations = []
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                if (i < len(labels) and j < len(labels)):
                    annotations.append(
                        {
                            "x": labels[i],
                            "y": labels[j],
                            "font": {"color": "white"},
                            "text": str(round(value / cm_sum, 3)) + '%',
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False,
                        }
                    )
        layout = {
            "title": title,
            "xaxis": {"title": "Predicted value"},
            "yaxis": {"title": "Real value"},
            "annotations": annotations
        }
        fig = go.Figure(data=data, layout=layout)
        return fig

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in physical_allowed_extensions

    @staticmethod
    def upload_data_files(folderfiles, mapfile):
        try:
            # Upload data files
            for file in folderfiles:
                if file and Helper.allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(data_files_folder, filename))

            # upload map file
            filePath = os.path.join(data_files_folder, secure_filename(mapfile.filename))
            mapfile.save(filePath)
            return 0

        except Exception as e:
            return e

    def create_FTP_conn(self, conn_details):
        ftp_conn = ftplib.FTP(conn_details['host'], conn_details['username'], conn_details['password'])
        print(ftp_conn.getwelcome())
        return ftp_conn

    def list_ftp_dirs(self, conn_details):
        ftp_conn = self.create_FTP_conn(conn_details)
        folders = []
        try:
            folders = ftp_conn.nlst()
        except ftplib.error_perm as resp:
            if str(resp) == "550 No files found":
                print("No files in this directory")
            else:
                raise
        return np.array(folders)

    @staticmethod
    def remove_empty_columns(filePath):
        data = pd.read_csv(filePath, sep=',', encoding='latin1')
        data = data.dropna(axis=1, how='all')
        data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        data.to_csv(filePath, index=False)
        data = pd.read_csv(filePath)

        return data

    @staticmethod
    def display_property(property_key):
        """
        Display the value of provided property key from the properties file
        @param property_key: property's key
        @return: value of the provided property's key
        """
        split_property = property_key.split('.')
        return config_parser.get(split_property[0], property_key)


    @staticmethod
    def generate_model_id():
        model_id = random.randrange(10000000000000, 99999999999999, 14)
        return model_id

    @staticmethod
    def validate_datestring(date_text):  # to be added to ent version
        try:
            datetime.datetime.strptime(date_text, '%d/%m/%Y %I:%M %p')
            return True
        except Exception as e:
            try:
                datetime.datetime.strptime(date_text, '%Y-%m-%d %H:%M:%S')
                return True
            except Exception as e:
                return False
            return False

    @staticmethod
    def validate_datestring_arr(date_arr):  # to be added to ent version
        try:
            datetime.datetime.strptime(date_arr[0], '%d/%m/%Y %I:%M:%S %p')
            return True
        except Exception as e:
            try:
                datetime.datetime.strptime(date_arr[0], '%Y-%m-%d %H:%M:%S')
                return True
            except Exception as e:
                return False
            return False

    @staticmethod
    def validate_timestring(time_text):  # to be added to ent version
        try:
            datetime.datetime.strptime(time_text, '%H:%M:%S')
            return 1
        except Exception as e:
            try:
                datetime.datetime.strptime(time_text, '%H:%M:%S')
                return 1
            except Exception as e:
                return 0
            return 0

    @staticmethod
    def validate_timestring_arr(time_arr):  # to be added to ent version
        try:
            datetime.datetime.strptime(time_arr[0], '%H:%M:%S')
            return 1
        except Exception as e:
            try:
                datetime.datetime.strptime(time_arr[0], '%H:%M:%S')
                return 1
            except Exception as e:
                return 0
            return 0

    @staticmethod
    def is_time(column_name):  # to be added to ent version
        is_date = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.is_date).filter(
                ModelEncodedColumns.column_name == column_name).first())
        is_date = is_date.flatten()
        return is_date

    @staticmethod
    def convert_time_to_seconds(interval):
        interval_periods = interval.split(':')

        return (int(interval_periods[0]) * 3600) + (int(interval_periods[1]) * 60) + int(interval_periods[2])

    @staticmethod
    def deletefolderfiles(*argv):
        try:
            for arg in argv:
                if os.path.isdir(arg):
                    files_in_directory = os.listdir(arg)
                    filtered_files = [file for file in files_in_directory if not file.endswith(".gitkeep")]
                    if (len(filtered_files) != 0):
                        for f in filtered_files:
                            # os.remove(os.path.join(arg, f))
                            path_to_file = os.path.join(arg, f)
                            os.remove(path_to_file)
            return 1
        except Exception as e:
            print('Ohh -delete_model_files...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def deleteobjectdetectionfiles(model_id):
        try:
            files_in_directory = os.listdir(app_results_path)
            filtered_files = [file for file in files_in_directory if (not (file.endswith(".gitkeep")) and (file.startswith(model_id)))]
            if (len(filtered_files) != 0):
                for f in filtered_files:
                    path_to_file = os.path.join(app_results_path, f)
                    os.remove(path_to_file)
            return 1
        except Exception as e:
            print('Ohh -delete_model_files...Something went wrong.')
            print(e)
            return 0

    def uploadfiles(self, uploadfolder, files):
        try:
            # Iterate for each file in the files List, and Save them
            for file in files:
                file.save(os.path.join(uploadfolder, file.filename))

            return "<h1>Files Uploaded Successfully.!</h1>"

        except Exception as e:
            print('Ohh -Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def get_lookup_value(lookup_id):
        lookup_value = ModelLookupTable.query.with_entities(ModelLookupTable.value).filter_by(key=lookup_id).first()
        return lookup_value['value']

    @staticmethod
    def get_lookup_key(lookup_key):
        lookup_key = ModelLookupTable.query.with_entities(ModelLookupTable.key).filter_by(value=lookup_key).first()
        return lookup_key['key']
