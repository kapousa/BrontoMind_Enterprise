import csv
import os
import pickle
import shutil
import string
from collections import defaultdict
from csv import DictReader
from os import listdir
from os.path import isfile, join
from random import shuffle
from app import db, config_parser
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler

from app.base.constants.BM_CONSTANTS import html_plots_location, html_short_path, app_root_path, \
    data_files_folder, df_location, scalars_location, dep_path, pkls_location
from app.base.db_models.ModelProfile import ModelProfile
from bm.utiles.Helper import Helper

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5


class ControllersHelper:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def plot_classification_report(model_name, classificationReport, title='Classification report ',
                                   with_avg_total=False, cmap=plt.cm.Blues):
        print(classificationReport)
        sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
        df = pd.DataFrame(classificationReport).iloc[:-1, :].T
        # fig = px.imshow(df)
        fig = px.imshow(df, labels=dict(x="Values", y="Class contribution", color="Importance"), x=df.columns,
                        y=df.index)
        fig.update_xaxes(side="top")
        # fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plotly.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    def plot_confusion_matrix(model_name, confusionmatrix, categories=[], title='Classification report'):
        categories = numpy.array(categories).flatten()
        print(confusionmatrix)
        df = pd.DataFrame(confusionmatrix)
        sns.heatmap(df, annot=True)

        # fig = px.imshow(df)
        fig = px.imshow(df, labels=dict(x="Predicted Class", y="Actual Class", color="Importance"), x=categories,
                        y=categories, text_auto=True)
        fig.update_xaxes(side="top")
        # fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plotly.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    def plot_features_importances_(model_name, features_column, importances):
        df = pd.DataFrame({'category': features_column, 'importance': importances})
        fig = px.bar(df, x='importance', y='category', color='importance')
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plt.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    def create_csv_data_file(self, output_csv_file_name: 'data.csv', header: ['label', 'file_name', 'text'],
                             req_extensions):
        csv_folder_location = '%s%s' % (app_root_path, data_files_folder)
        csv_file_location = '%s%s%s' % (app_root_path, data_files_folder, output_csv_file_name)
        data = self.create_txt_data_file(csv_folder_location, req_extensions)
        with open(csv_file_location, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerows(data)

            return 1

    @staticmethod
    def get_folder_structure(self, path_of_the_directory, req_extensions=('.txt')):
        try:
            full_path = path_of_the_directory
            folders_list = [f for f in listdir(full_path) if
                            isfile(join(full_path, f)) == False]  # get sub folders list
            ext = req_extensions  # ex: ('txt', 'docx')
            files_list = []
            folder_structure = dict()

            # for i in folders_list:
            #     sub_folder_path = full_path + '/' + i
            #     dictionary_fields = []
            #     for file_name in os.listdir(sub_folder_path):
            #         if file_name.endswith(ext):
            #             dictionary_fields.append(file_name)
            #         else:
            #             continue
            #     dic_keys = dictionary_fields
            #     dic_values = dictionary_fields
            #     folder_structure.update({i: dict(zip(dic_keys, dic_values))})
            return folders_list  # , folder_structure
        except  Exception as e:
            print(e)
            return 0

    def create_txt_data_file(self, path_of_the_directory, req_extensions=('.txt')):
        try:
            full_path = path_of_the_directory
            folders_list = [f for f in listdir(full_path) if
                            isfile(join(full_path, f)) == False]  # get sub folders list
            ext = req_extensions  # ex: ('txt', 'docx')
            data_list = []

            for i in folders_list:
                sub_folder_path = full_path + '/' + i
                dictionary_fields = []
                for file_name in os.listdir(sub_folder_path):
                    if file_name.endswith(ext):
                        with open(sub_folder_path + '/' + file_name, 'rb') as file:
                            file_text = file.readline().decode(errors='replace').replace('/n', '')
                            data_list.append([i, file_text.strip()])
                    else:
                        continue
            return data_list

        except  Exception as e:
            print(e)
            return 0

    def print_frequency_dist_(self, docs):
        try:
            tokens = defaultdict(list)
            most_common = []
            categories = []
            for doc in docs:
                doc_label = doc[0]
                doc_text = doc[1]
                doc_tokens = word_tokenize(doc_text)
                tokens[doc_label].extend(doc_tokens)

            for category_label, category_tokens in tokens.items():
                print(category_label)
                fd = FreqDist(category_tokens)
                most_common_3 = fd.most_common(3)
                categories.append(category_label)
                most_common.append(str(most_common_3))

            return categories, most_common
        except Exception as e:
            print(e)
            return 0

    def print_frequency_dist(self, docs):
        try:
            tokens = defaultdict(list)
            most_common = []
            categories = []
            for doc in docs:
                doc_label = doc[0:-1]
                doc_text = doc[-1]
                doc_tokens = word_tokenize(doc_text)
                tokens[doc_label].extend(doc_tokens)

            for category_label, category_tokens in tokens.items():
                print(category_label)
                fd = FreqDist(category_tokens)
                most_common_3 = fd.most_common(3)
                categories.append(category_label)
                most_common.append(str(most_common_3))

            return categories, most_common
        except Exception as e:
            print(e)
            return 0

    def create_FTP_data_set(self, location_details, labels, model_id=0):
        try:
            output_file = '%s%s%s' % (df_location, str(model_id), '.txt')
            helper = Helper()
            ftp_conn = helper.create_FTP_conn(location_details)
            with open(output_file, 'w', encoding='utf8') as outfile:
                for label in labels:
                    current_folder = "%s%s" % ("/", label)
                    ftp_conn.cwd(current_folder)
                    files_list = ftp_conn.nlst()
                    for filename in files_list:
                        fullfilename = filename
                        gFile = open("temp.txt", "wb")
                        ftp_conn.retrbinary(f"RETR {fullfilename}", gFile.write)
                        gFile.close()
                        with open("temp.txt", 'rb') as file:
                            text = file.read().decode(errors='replace').replace('\n', '')
                        outfile.write('%s\t%s\t%s\n' % (label, fullfilename, text))
                        gFile.close()
                ftp_conn.quit()
            outfile.close()

            return 1
        except  Exception as e:
            print(e)
            return 0

    def setup_docs(self, full_file_path):
        try:
            docs = []

            with open(full_file_path, 'r', encoding='utf8') as datafile:
                for row in datafile:
                    parts = np.array(row.split('\t'))
                    # if (len(parts) >= 2):
                    #     doc = (parts[0], parts[1].strip())
                    #     docs.append(doc)
                    if (len(parts) >= 2):
                        txt = parts[-1]
                        cats = parts[0:len(parts) - 1]
                        doc = (*cats, txt.strip())
                        docs.append(doc)
                return docs
        except  Exception as e:
            print(e)
            return 0

    def get_tokens(self, text):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if not t in stopwords]
        return tokens

    def clean_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    @staticmethod
    def scale_data(data, model_file_name, encode_df_testing_values):
        """
        Adjust dataframe to standard scale
        @param data: original dataframe
        @return: data_transformed - scaled dataframe
        """
        # Sclaing testing values
        scalar_file_name = scalars_location + model_file_name + '_scalear.sav'
        s_c = pickle.load(open(scalar_file_name, 'rb'))
        test_x = s_c.transform(encode_df_testing_values)
        mms = MinMaxScaler()
        mms.fit(data)
        data_transformed = mms.transform(data)

        return data_transformed

    @staticmethod
    def model_deployed(model_id):
        deployemnt_statu = ModelProfile.query.with_entities(ModelProfile.deployed).filter_by(model_id=model_id).first()
        deployemnt_statu = numpy.array(deployemnt_statu).flatten()
        dep_value = deployemnt_statu[0]

        if (str(dep_value) == config_parser.get('DeploymentStatus', 'DeploymentStatus.notdeployed')):
            return False
        else:
            return True