import os
import pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly as pl
import plotly.graph_objs as pgo
from nltk.corpus import stopwords
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from app import config_parser
from app.base.constants.BM_CONSTANTS import html_plots_location, html_short_path, df_location, clusters_keywords_file, \
    output_docs_location, labeled_data_filename
from bm.utiles.Helper import Helper


class ClusteringControllerHelper:
    # stop_words = set(stopwords.words('english'))

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.stop_words = set(stopwords.words('english'))

    def plot_clustering_report_(self, df, model, label, model_name='file_name'):
        # Represent neighborhoods as in previous bubble chart, adding cluster information under color.
        df = pd.DataFrame(df)
        df['label'] = label

        x_axis = np.array(df[0])
        y_axis = np.array(df[1])
        x_center = numpy.array(model.cluster_centers_[:, 0])
        y_center = numpy.array(model.cluster_centers_[:, 1])
        u_label = np.unique(label)

        fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.scatter(x_axis, y_axis, s=50)
        plt.scatter(x_center, y_center, marker='x')
        fig.show()
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': True}, auto_open=False)

        return html_path

    @staticmethod
    def plot_clustering_report(df, model, label, model_name='file_name'):
        # Represent neighborhoods as in previous bubble chart, adding cluster information under color.
        df = pd.DataFrame(df)
        df['label'] = label
        u_label = np.unique(label)
        trace_arr = []
        for i in u_label:
            label_data = df.loc[df['label'] == i]
            trace0 = pl.graph_objs.Scatter(x=label_data[0],
                                           y=label_data[1],
                                           mode="markers",
                                           name="Cluster:" + str(i),
                                           text="Cluster:" + str(i),
                                           marker=pgo.scatter.Marker(symbol='x',
                                                                     size=12,
                                                                     color=u_label, opacity=0.5),
                                           showlegend=True
                                           )
            trace_arr.append(trace0)

        # Represent cluster centers.
        trace1 = pl.graph_objs.Scatter(x=model.cluster_centers_[:, 0],
                                       y=model.cluster_centers_[:, 1],
                                       name='Center of the cluster',
                                       mode='markers',
                                       marker=pgo.scatter.Marker(symbol='x',
                                                                 size=12,
                                                                 color='black'),
                                       showlegend=True
                                       )
        trace_arr.append(trace1)

        layout5 = pgo.Layout(title='Data Clusters',
                             xaxis=pgo.layout.XAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             yaxis=pgo.layout.YAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             hovermode='closest')

        data7 = pgo.Data(trace_arr)
        layout7 = layout5
        layout7['title'] = 'Data Clusters'
        fig7 = pgo.Figure(data=data7, layout=layout7)

        # Plot model
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        fig = make_subplots(rows=2, cols=1)
        # 1- clusters
        for k in range(len(fig7.data)):
            fig.add_trace(fig7.data[k], row=1, col=1)
        # 2- Elbow graph
        # TODO: Add code to implent Elbow function and plot it with the clustering graph

        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    @staticmethod
    def plot_data_points(df, model, label, model_name='file_name'):
        # Represent neighborhoods as in previous bubble chart, adding cluster information under color.
        df = pd.DataFrame(df)
        df['label'] = label
        u_label = np.unique(label)
        trace_arr = []
        for i in u_label:
            label_data = df.loc[df['label'] == i]
            trace0 = pl.graph_objs.Scatter(x=label_data[0],
                                           y=label_data[1],
                                           mode="markers",
                                           name="Cluster:" + str(i),
                                           text="Cluster:" + str(i),
                                           marker=pgo.scatter.Marker(symbol='x',
                                                                     size=12,
                                                                     color=u_label, opacity=0.5),
                                           showlegend=True
                                           )
            trace_arr.append(trace0)

        # Represent cluster centers.
        trace1 = pl.graph_objs.Scatter(x=model.cluster_centers_[:, 0],
                                       y=model.cluster_centers_[:, 1],
                                       name='Center of the cluster',
                                       mode='markers',
                                       marker=pgo.scatter.Marker(symbol='x',
                                                                 size=12,
                                                                 color='black'),
                                       showlegend=True
                                       )
        trace_arr.append(trace1)

        layout5 = pgo.Layout(title='Data Clusters',
                             xaxis=pgo.layout.XAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             yaxis=pgo.layout.YAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             hovermode='closest')

        data7 = pgo.Data(trace_arr)
        layout7 = layout5
        layout7['title'] = 'Data Clusters'
        fig7 = pgo.Figure(data=data7, layout=layout7)

        # Plot model
        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        fig = make_subplots(rows=2, cols=1)
        # 1- clusters
        for k in range(len(fig7.data)):
            fig.add_trace(fig7.data[k], row=1, col=1)
        # 2- Elbow graph
        # TODO: Add code to implent Elbow function and plot it with the clustering graph

        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    @staticmethod
    def plot_elbow_graph(data, model_name='file_name'):
        data = np.reshape(data, (len(data), 1))
        mms = MinMaxScaler()
        mms.fit(data)
        data_transformed = mms.transform(data)

        Sum_of_squared_distances = []
        K = range(1, 15)
        x_axis = [*K]
        y_axis = Sum_of_squared_distances
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(data_transformed)
            Sum_of_squared_distances.append(km.inertia_)

        fig = pgo.Figure(
            data=[pgo.Scatter(x=x_axis, y=y_axis, line_color="crimson", marker=pgo.scatter.Marker(symbol='x',
                                                                                                  size=10,
                                                                                                  color='black'),
                              name="Elbow Graph",
                              text="Number of Clusters",
                              showlegend=True)],
            layout_title_text="A Graph of suggested number of clusters"
        )

        html_file_location = html_plots_location + model_name + ".html"
        html_path = html_short_path + model_name + ".html"
        pl.offline.plot(fig, filename=html_file_location, config={'displayModeBar': False}, auto_open=False)

        return html_path

    @staticmethod
    def extract_clusters_keywords(model, k, vectorizer):
        """
            Extract the keywords of each cluster then save the results in pkl file
        """
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        cluster_keywords = {}
        number_of_keywords = int(
            config_parser.get('SystemConfigurations', 'SystemConfigurations.number_of_clustering_keywords'))
        for i in range(k):
            cluster_name = "Cluster_" + str(i)
            cluster_terms = ''
            for j in order_centroids[i, :number_of_keywords]:  # print out 10 feature terms of each cluster
                cluster_terms = cluster_terms + terms[j] + ('' if (j == number_of_keywords) else ', ')
            cluster_keywords[cluster_name] = cluster_terms

        # Save clusters' keywords in pkle file
        a_file = open(clusters_keywords_file, "wb")
        pickle.dump(cluster_keywords, a_file)
        a_file.close()

        return cluster_keywords

    @staticmethod
    def get_clustering_keywords():
        with open(clusters_keywords_file, 'rb') as f:
            loaded_clustering_keywords = pickle.load(f)

        return loaded_clustering_keywords

    @staticmethod
    def get_cluster_keywords(cluster):
        """
        Return arra of keywords of provided cluster
        @param cluster:
        @return:
        """
        clusters_keywords = ClusteringControllerHelper.get_clustering_keywords()
        cluster_keywords = clusters_keywords['Cluster_' + str(cluster)]

        return cluster_keywords

    def evaluate_clustfer(self, classifier, vectorizer, X_test, y_test, categories):
        return 0, 1, 2

    def get_clusterfer_splits(self, docs):
        shuffle(docs)
        return docs

    def create_clustering_data_set(self, files_path):
        try:
            output_file = '%s%s' % (files_path, 'data.pkl')
            files_data = {}
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, 'w', encoding='utf8') as outfile:
                for filename in os.listdir(files_path):
                    with open(filename, 'rb') as file:
                        text = file.read().decode(errors='replace').replace('\n', '')
                        files_data.append(text)
                        outfile.write('%s\n' % (text))
            df = pd.DataFrame(files_data)
            df.to_pickle(outfile)

            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_clustering_csv_data_set(self, csv_file_path, features_list):
        try:
            df = pd.read_csv(csv_file_path)
            df = df.loc[:, features_list]
            output_file = '%s%s' % (df_location, 'data.pkl')
            df.to_pickle(output_file)

            return 1
        except  Exception as e:
            print(e)
            return 0

    def create_clustering_FTP_data_set(self, location_details):
        try:
            output_file = '%s%s' % (df_location, 'data.pkl')
            helper = Helper()
            ftp_conn = helper.create_FTP_conn(location_details)
            files_data = []

            # Reading files contents and save it in pkl file
            ftp_conn.cwd(df_location)
            files_list = ftp_conn.nlst()
            for filename in files_list:
                fullfilename = filename
                gFile = open("temp.txt", "wb")
                ftp_conn.retrbinary(f"RETR {fullfilename}", gFile.write)
                gFile.close()
                with open("temp.txt", 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    files_data.append(text)
                gFile.close()
            ftp_conn.quit()

            df = pd.DataFrame(files_data)
            df.to_pickle(output_file)

            return 1
        except  Exception as e:
            print(e)
            return 0

    @staticmethod
    def generate_labeled_datafile(file_name, labels: []):
        """
        Create csv file with label column from data file of unlabeled data
        @param file_name:
        @param labels:
        @return: path of updated data file
        """
        try:
            data_file_location = "%s%s" % (df_location, file_name)
            updated_data_file_location = "%s%s" % (output_docs_location, labeled_data_filename)

            df = pd.read_csv(data_file_location)
            df['label'] = labels
            df.to_csv(updated_data_file_location, index=False)

            return updated_data_file_location
        except Exception as e:
            return Helper.display_property('ErrorMessages.fail_create_updated_data_file')
