# from itertools import izip
import csv
import os
import pickle
import string
from collections import defaultdict
from csv import DictReader
from os import listdir
from os.path import isfile, join
from random import shuffle

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

from app.base.constants.BM_CONSTANTS import html_plots_location, html_short_path, app_root_path, \
    pkls_location, \
    data_files_folder, df_location
from bm.utiles.Helper import Helper


class ClassificationControllerHelper:
    # stop_words = set(stopwords.words('english'))

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
        fig = px.imshow(df,
                        labels=dict(x="Values", y="Class contribution", color="Importance"),
                        x=df.columns,
                        y=df.index
                        )
        fig.update_xaxes(side="top")
        # fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plotly.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    @staticmethod
    def plot_confusion_matrix(model_name, confusionmatrix, categories=[], title='Classification report'):
        categories = numpy.array(categories).flatten()
        print(confusionmatrix)
        df = pd.DataFrame(confusionmatrix)
        sns.heatmap(df, annot=True)

        # fig = px.imshow(df)
        fig = px.imshow(df,
                        labels=dict(x="Predicted Class", y="Actual Class", color="Importance"),
                        x=categories,
                        y=categories,
                        text_auto=True
                        )
        fig.update_xaxes(side="top")
        # fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plotly.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    @staticmethod
    def plot_features_importances_(model_name, features_column, importances):
        df = pd.DataFrame({'category': features_column, 'importance': importances})
        fig = px.bar(df, x='importance', y='category', color='importance')
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        plt.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    def create_csv_data_file(self, output_csv_file_name: 'data.csv',
                             header: ['label', 'file_name', 'text'],
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

    def create_classification_data_set(self, files_path, labels):
        try:
            output_file = '%s%s' % (files_path, 'data.txt')
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, 'w', encoding='utf8') as outfile:
                for label in labels:
                    dir = '%s%s' % (files_path, label)
                    for filename in os.listdir(dir):
                        fullfilename = '%s%s%s' % (dir, '/', filename)
                        with open(fullfilename, 'rb') as file:
                            text = file.read().decode(errors='replace').replace('\n', '')
                            outfile.write('%s\t%s\n' % (label, text))
            outfile.close()
            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_classification_csv_data_set(self, csv_file_path):
        try:
            output_file = '%s%s' % (df_location, 'data.txt')
            if os.path.exists(output_file):
                os.remove(output_file)
            # Open file
            with open(output_file, 'w', encoding='utf8') as outfile:
                # Create reader object by passing the file
                # object to reader method
                with open(csv_file_path, 'r') as read_obj:
                    csv_dict_reader = DictReader(read_obj)
                    for row in csv_dict_reader:
                        aa = row
                        # print(row)
                        data_row = []
                        for key, value in row.items():
                            data_row.append(value)
                        if (data_row[0] != '' and data_row[1] != ''):
                            outfile.write('%s\t' % (data_row[1]))
                            text = data_row[0].replace('\n', '')
                            outfile.write('%s\n' % (text))
            outfile.close()

            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_FTP_data_set(self, location_details, labels):
        try:
            output_file = '%s%s' % (df_location, 'data.txt')
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

    def get_classification_splits(self, docs):
        shuffle(docs)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        # pivot = int(0.8 * len(docs))
        pivot = int(1 * len(docs))

        for i in range(0, pivot):
            X_train.append(docs[i][1])
            y_train.append(docs[i][0])

        for i in range(0, len(docs)):
            X_test.append(docs[i][1])
            y_test.append(docs[i][0])

        return X_train, X_test, y_train, y_test

    def train_classifier(self, docs, categories):
        try:
            X_train, X_test, y_train, y_test = self.get_classification_splits(docs)

            vectorized = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

            # Create doc-term matrix
            dtm = vectorized.fit_transform(X_train)

            # Train the model
            naive_bays_classifier = MultinomialNB().fit(dtm, y_train)

            # Model evaluations
            train_precision, train_recall, train_f1 = self.evaluate_classifier(naive_bays_classifier, vectorized,
                                                                               X_train, y_train, categories)
            test_precision, test_recall, test_f1 = self.evaluate_classifier(naive_bays_classifier, vectorized, X_test,
                                                                            y_test, categories)

            # store the classifier
            clf_filename = '%s%s%s' % (app_root_path, pkls_location, 'classifier_pkl.pkl')
            pickle.dump(naive_bays_classifier, open(clf_filename, 'wb'))

            # Store vectorized
            vic_filename = '%s%s%s' % (app_root_path, pkls_location, 'vectorized_pkl.pkl')
            pickle.dump(vectorized, open(vic_filename, 'wb'))

            precision = numpy.array([train_precision, test_precision])
            recall = numpy.array([train_recall, test_recall])
            f1 = numpy.array([train_f1, test_f1])

            all_return_value = {'train_precision': str(precision),
                                'train_recall': str(recall),
                                'train_f1': str(f1),
                                'test_precision': str(precision),
                                'test_recall': str(recall),
                                'test_f1': str(f1)}

            # all_return_value = {'train_precision': '',
            #                     'train_recall': '',
            #                     'train_f1': '',
            #                     'test_precision': '',
            #                     'test_recall': '',
            #                     'test_f1': ''}

            return all_return_value

        except  Exception as e:
            print(e)
            return e

    def evaluate_classifier(self, classifier, vectorizer, X_test, y_test, categories):
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_tfidf)
        precision = precision_score(y_test, y_pred,
                                    pos_label='positive',
                                    average=None)
        recall = recall_score(y_test, y_pred,
                              pos_label='positive',
                              average=None)
        f1 = f1_score(y_test, y_pred,
                      pos_label='positive',
                      average=None)
        cm = confusion_matrix(y_test, y_pred)

        classificationreport = classification_report(y_test, y_pred,
                                                     # labels=labels,
                                                     # target_names=target_names,
                                                     output_dict=True)
        # html_path = ClassificationControllerHelper.plot_features_importances_(file_name, t_columns, importances)
        html_path = ClassificationControllerHelper.plot_confusion_matrix('file_name', cm, categories)

        print(("%s\t%s\t%s\n") % (precision, recall, f1))

        return precision, recall, f1

    def classify(self, text):
        # Load model
        clf_filename = '%s%s%s' % (app_root_path, pkls_location, '/classifier_pkl.pkl')
        np_clf = pickle.load(open(clf_filename, 'rb'))

        # load vectorizer
        vec_filename = '%s%s%s' % (app_root_path, pkls_location, '/vectorized_pkl.pkl')
        vectorizer = pickle.load(open(vec_filename, 'rb'))

        pred = np_clf.predict(vectorizer.transform([text]))

        return pred
