import uuid

from sqlalchemy import and_

from app.base.db_models.ModelFeatures import ModelFeatures
from app.base.db_models.ModelForecastingResults import ModelForecastingResults
from app.base.db_models.ModelProfile import ModelProfile
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from app.base.db_models.ModelAPIModelMethods import ModelAPIModelMethods
from app import db
from app.base.db_models.ModelLabels import ModelLabels
import numpy
import random

from app.authentication.models import Users
from base.db_models.ModelLookupTables import ModelLookupTable


def add_features(model_id, features_list):
    features_data = []
    # Add model profile to the database
    for i in features_list:
        data_item = {'feature_name': i,
                     'model_id': model_id}
        features_data.append(data_item)
    # Delete current features
    # ModelFeatures.query.filter().delete()
    # db.session.commit()
    db.session.bulk_insert_mappings(ModelFeatures, features_data)
    db.session.commit()

    return 1


def add_labels(model_id, labels_list):
    # Add model profile to the database
    label_data = []
    # Add model profile to the database
    for i in labels_list:
        data_item = {'label_name': i,
                     'model_id': model_id}
        label_data.append(data_item)
    # Delete current features
    # aa = ModelLabels.query.filter().delete()
    # db.session.commit()
    db.session.bulk_insert_mappings(ModelLabels, label_data)
    db.session.commit()

    return 1


def add_forecasting_results(model_id, actual, predicted, period_dates):
    forecasting_result_data = []
    # Add model timeforecasting result to the database
    for i in range(len(period_dates)):
        data_item = {'actual': actual[i],
                     'predicted': predicted[i],
                     'period_dates': str(period_dates[i]),
                     'model_id': model_id}
        forecasting_result_data.append(data_item)
    # Delete current timeforecasting result
    # ModelForecastingResults.query.filter().delete()
    # db.session.commit()
    db.session.bulk_insert_mappings(ModelForecastingResults, forecasting_result_data)
    db.session.commit()

    return 1


def add_api_details(model_id, api_details_id, api_version):
    # Add model profile to the database
    api_details_data = []
    key = uuid.uuid1()
    data_item = {'api_details_id': api_details_id,
                 'api_version': api_version,
                 'private_key': str(key.hex),
                 'public_key': str(key.int),
                 'model_id': model_id}
    api_details_data.append(data_item)
    # Delete current features
    # aa = ModelAPIDetails.query.filter().delete()
    # db.session.commit()
    db.session.bulk_insert_mappings(ModelAPIDetails, api_details_data)
    db.session.commit()
    return 1


def get_model_id(model_name):
    model_id = numpy.array(ModelProfile.query.with_entities(ModelProfile.model_id).filter(ModelProfile.model_name == model_name).first())
    flatten_model_id = model_id.flatten()
    return flatten_model_id[0]


def get_features(model_id=0):
    model_features = numpy.array(ModelFeatures.query.with_entities(ModelFeatures.feature_name).filter(ModelFeatures.model_id == str(model_id)).all())
    flatten_model_features = model_features.flatten()
    return flatten_model_features


def get_labels(model_id=0):
    model_labels = numpy.array(ModelLabels.query.with_entities(ModelLabels.label_name).filter(ModelLabels.model_id == str(model_id)).all())
    flatten_model_labels = model_labels.flatten()
    return flatten_model_labels


def get_api_details():
    model_api_details = numpy.array(
        ModelAPIDetails.query.with_with_entities(ModelAPIDetails.api_version, ModelAPIDetails.public_key,
                                                 ModelAPIDetails.private_key).all())
    flatten_model_api_details = model_api_details.flatten()
    return flatten_model_api_details


def getmodelencodedcolumns(model_id, column_type):
    model_encoded_columns =ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(and_(ModelEncodedColumns.column_type == column_type, ModelEncodedColumns.model_id == model_id)).all()
    model_encoded_columns = numpy.array(model_encoded_columns)
    flatten_model_encoded_column = model_encoded_columns.flatten()
    return flatten_model_encoded_column


def get_encoded_labels(model_id, column_type):
    model_encoded_columns = numpy.array(ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(and_(ModelEncodedColumns.column_type == column_type, ModelEncodedColumns.model_id == model_id)).all())
    flatten_model_encoded_column = model_encoded_columns.flatten()
    return flatten_model_encoded_column


def get_model_name(model_id):
    model_name = numpy.array(ModelProfile.query.with_entities(ModelProfile.model_name).filter_by(model_id = model_id).first())
    flatten_model_name = model_name.flatten()
    return flatten_model_name[0]


def add_encoded_column_values(model_id, column_name, column_values, column_type):
    try:
        encode_prefix = []
        for i in column_values:
            data_item = {'column_name': i,
                         'column_type': column_type,
                         'model_id': model_id}
            encode_prefix.append(data_item)
        db.session.bulk_insert_mappings(ModelEncodedColumns, encode_prefix)
        db.session.commit()
        print('Encoded column values added successfully.')
        return 1
    except  Exception as e:
        print('Ohh -add_encoded_column_values...Something went wrong.')
        print(e)
        return 0


def delete_encoded_columns():
    aa = ModelEncodedColumns.query.filter().delete()
    db.session.commit()
    return 1


def encode_testing_features_values(model_id, testing_values: dict):
    try:
        returned_encoded_features = []
        for i in testing_values.keys():
            encoded_features = db.session.query(ModelEncodedColumns.column_name).filter(
                ModelEncodedColumns.column_name.contains(i)).all()
            if len(encoded_features) != 0:
                encoded_column = testing_values.get(i)
                for j in range(len(encoded_features)):
                    encoded_features_str = str(encoded_features[j])
                    if encoded_column in encoded_features_str:
                        returned_encoded_features.append(1)
                    else:
                        returned_encoded_features.append(0)
            else:
                returned_encoded_features.append(testing_values.get(i))
        return returned_encoded_features
    except  Exception as e:
        print('Ohh -encode_labels_values...Something went wrong.')
        print(e)
        return 0


def update_api_details_id(api_details_id):
    # num_rows_updated = ModelAPIModelMethods.query.update(dict(api_details_id=api_details_id))
    # db.session.commit()
    #
    #
    # #db.close_all_sessions
    return 1


def get_lookup_value(lookup_key):
    lookup_value = numpy.array(ModelLookupTable.query.with_entities(ModelLookupTable.value).filter_by(key =lookup_key).first())
    flatten_lookup_value = lookup_value.flatten()
    return flatten_lookup_value[0]


def get_lookup_key(lookup_key):
    lookup_value = numpy.array(ModelLookupTable.query.with_entities(ModelLookupTable.value).filter_by(key =lookup_key).first())
    flatten_lookup_value = lookup_value.flatten()
    return flatten_lookup_value[0]


def get_user_fullname(user_id):
    user_full_name = numpy.array(Users.query.with_entities(Users.first_name, Users.last_name).filter_by(id =user_id).first())
    full_name = "%s%s%s" % (user_full_name[0]," ",user_full_name[1])
    return full_name