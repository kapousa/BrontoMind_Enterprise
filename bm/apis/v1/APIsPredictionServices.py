import json
import numpy

from bm.controllers.prediction.ModelController import predict_values_from_model
from bm.db_helper.AttributesHelper import get_features, get_model_name, get_labels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


def predictvalues(model_id, content):
    model_name = get_model_name(model_id)
    features_list = get_features(model_id)
    lables_list = get_labels(model_id)
    testing_values = []
    for i in features_list:
        feature_value = str(content[i])
        final_feature_value = feature_value # float(feature_value) if feature_value.isnumeric() else feature_value
        testing_values.append(final_feature_value)
    predicted_value = predict_values_from_model(model_name, testing_values)

    # Create predicted values json object
    predicted_values_json = {}
    for j in range(len(predicted_value)):
        for i in range(len(lables_list)):
            bb =  predicted_value[j][i]
            predicted_values_json[lables_list[i]] = predicted_value[j][i]
            # NpEncoder = NpEncoder(json.JSONEncoder)
        json_data = json.dumps(predicted_values_json, cls=NpEncoder)


    return json_data

def getplotiamge(content):
    return 0

def getmodelfeatures():
    features_list = get_features(0)
    features_json = {}
    j = 0
    for i in features_list:
        yy = str(i)
        features_json[i] = i
        j += 1
    # NpEncoder = NpEncoder(json.JSONEncoder)
    json_data = json.dumps(features_json, cls=NpEncoder)

    return json_data

def getmodellabels():
    labels_list = get_labels()
    labelss_json = {}
    j = 0
    for i in labels_list:
        yy = str(i)
        labelss_json[i] = i
        j += 1
    # NpEncoder = NpEncoder(json.JSONEncoder)
    json_data = json.dumps(labelss_json, cls=NpEncoder)

    return json_data

def getmodelprofile(contents):
    return 0

def nomodelfound():
    no_model_found = {'no_model':'No Model found' }
    json_data = json.dumps(no_model_found, cls=NpEncoder)
    return json_data