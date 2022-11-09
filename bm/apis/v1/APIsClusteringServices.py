import json

import numpy

from bm.apis.v1.APIsPredictionServices import NpEncoder
from bm.controllers.classification.ClassificationController import ClassificationController
from bm.controllers.clustering.ClusteringController import ClusteringController
from bm.db_helper.AttributesHelper import get_model_name, get_features, get_labels


class APIsClusteringServices:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def cluster_data(self, content):
        testing_values = []
        features_list = get_features(0)
        for i in features_list:
            feature_value = str(content[i])
            final_feature_value = feature_value  # float(feature_value) if feature_value.isnumeric() else feature_value
            testing_values.append(final_feature_value)
        clustering_controller = ClusteringController()
        clusters_dic = clustering_controller.get_data_cluster(testing_values)

        # Create predicted values json object
        cluster_data_json = {}
        for k, v in clusters_dic.items():
            cluster_data_json[k] = v

        json_data = json.dumps(cluster_data_json, cls=NpEncoder)

        return json_data

    def cluster_data_list(self, content):
        cluster_data_json = {}
        for i in range(len(content)):
            class_item = self.cluster_data(content[i])
            cluster_data_json[i] = class_item
        json_data = json.dumps(cluster_data_json, cls=NpEncoder)

        return json_data
