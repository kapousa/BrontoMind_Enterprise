import os
import pathlib
import pickle
import random
import shutil
from datetime import datetime
from random import randint

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from flask import session
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app import db, config_parser
from app.base.constants.BM_CONSTANTS import plot_zip_locations, pkls_location, df_location, \
    plot_locations, scalars_location, image_short_path, data_files_folder, \
    app_root_path
from app.base.db_models.ModelProfile import ModelProfile
from bm.controllers.BaseController import BaseController
from bm.controllers.ControllersHelper import ControllersHelper
from bm.controllers.classification.ClassificationControllerHelper import ClassificationControllerHelper
from bm.core.ModelProcessor import ModelProcessor
from bm.datamanipulation.AdjustDataFrame import convert_data_to_sample
from bm.datamanipulation.AdjustDataFrame import remove_null_values
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import add_api_details, \
    update_api_details_id
from bm.db_helper.AttributesHelper import add_features, add_labels, delete_encoded_columns
from bm.utiles.CVSReader import get_only_file_name
from bm.utiles.CVSReader import getcvsheader, get_new_headers_list, reorder_csv_file
from bm.utiles.Helper import Helper

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5


class ClassificationController:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def predict_values_from_model(self, testing_value):
        try:
            classificationcontrollerHelper = ClassificationControllerHelper()
            class_name = classificationcontrollerHelper.classify(testing_value)

            return class_name

        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e.with_traceback())
            return [
                'Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset']

    def run_the_classification_model(self, root_path, csv_file_location, categories_column, classification_features,
                                     ds_source, ds_goal):
        """
            Currently this function is not used at any place
        """
        # ------------------Preparing data frame-------------------------#
        cvs_header = getcvsheader(csv_file_location)
        new_headers_list = get_new_headers_list(cvs_header, categories_column)
        reordered_data = reorder_csv_file(csv_file_location, new_headers_list)
        data = reordered_data  # pd.read_csv(csv_file_location)
        new_headers_list = np.append(classification_features, [categories_column])
        data = data[new_headers_list]
        model_id = Helper.generate_model_id()

        # Determine features and lables
        features_last_index = len(new_headers_list) - (len(categories_column))
        model_features = new_headers_list[0:features_last_index]
        model_labels = categories_column

        # 1-Clean the data frame
        data = remove_null_values(data)
        if (len(data) == 0):  # No data found after cleaning
            return 0

        dd = data.max(numeric_only=True)
        bb = data.describe()
        print(data.describe())

        # 2- Encode the data frame
        deleteencodedcolumns = delete_encoded_columns()

        data_column_count = len(data.columns)
        testing_values_len = data_column_count - len(categories_column)

        # take slice from the dataset, all rows, and cloumns from 0:8
        features_df = data[model_features]
        labels_df = data[model_labels]
        print(labels_df.describe())

        real_x = data.loc[:, model_features]
        real_y = data.loc[:, model_labels]
        dcp = DataCoderProcessor()
        real_x = dcp.vectrise_feature_text(model_id, real_x)
        real_y = dcp.encode_labels(model_id, real_y)
        encoded_data = pd.concat((real_x, real_y), axis=1, join='inner')
        # real_x = encode_one_hot(model_id, features_df, 'F')  # 2 param (test vales)
        # real_y = encode_one_hot(model_id, labels_df, 'L')  # (predict values)

        training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.15, random_state=0)

        # Add standard scalar
        s_c = StandardScaler(with_mean=False)  # test
        training_x = s_c.fit_transform(training_x)
        test_x = s_c.transform(testing_x)
        file_name = get_only_file_name(csv_file_location)
        scalar_file_name = scalars_location + str(model_id) + '_scalear.sav'
        pickle.dump(s_c, open(scalar_file_name, 'wb'))

        # Select proper model
        mp = ModelProcessor()
        cls = mp.classificationmodelselector(len(encoded_data))
        # cls = # LinearRegression() #MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), n_jobs=-1)  # KNeighborsRegressor)
        cls.fit(training_x, training_y)

        model_file_name = pkls_location + str(model_id) + '_model.pkl'
        pickle.dump(cls, open(model_file_name, 'wb'))
        y_pred = cls.predict(test_x)

        # Evaluating the Algorithm
        Mean_Absolute_Error = round(
            metrics.mean_absolute_error(numpy.array(testing_y, dtype=object), numpy.array(y_pred, dtype=object)),
            2)  # if not is_classification else 'N/A'
        Mean_Squared_Error = round(metrics.mean_squared_error(testing_y, y_pred),
                                   2)  # if not is_classification else 'N/A'
        Root_Mean_Squared_Error = round(np.sqrt(metrics.mean_squared_error(testing_y, y_pred)),
                                        2)  # if not is_classification else 'N/A'
        conf_matrix = confusion_matrix(testing_y, y_pred)
        print(conf_matrix)
        c_m = round(BaseController.get_cm_accurcy(conf_matrix), 2)
        acc = np.array(round(cls.score(training_x, training_y) * 100, 2))

        # Delete old visualization images
        dir = plot_locations
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        for f in os.listdir(plot_zip_locations):
            os.remove(os.path.join(plot_zip_locations, f))

        # Show prediction
        heatmap_image_path = os.path.join(plot_locations, str(model_id) + 'heatmap_plot.png')
        ax = sns.heatmap(conf_matrix / sum(conf_matrix), annot=True, fmt='.2%', cmap='Blues')
        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        ## Ticket labels - List must be in alphabetical order
        dcp = DataCoderProcessor()
        category_values_arr = testing_y[model_labels[0]].to_numpy()
        category_values_arr = category_values_arr.flatten()

        image_db_path = image_short_path + str(model_id) + 'heatmap_plot.png'
        t_columns = real_x.columns
        importances = numpy.array(cls.coef_).flatten()
        classificationreport = classification_report(testing_y, y_pred,
                                                     # labels=labels,
                                                     # target_names=target_names,
                                                     output_dict=True)
        # html_path = ClassificationControllerHelper.plot_features_importances_(file_name, t_columns, importances)
        html_path = ClassificationControllerHelper.plot_classification_report(file_name, classificationreport)

        for i in range(len(model_features)):
            for j in range(len(model_labels)):
                img_prefix = '_' + model_features[i] + '_' + model_labels[j]
                plot_image_path = os.path.join(plot_locations,
                                               str(model_id) + img_prefix + '_plot.png')
                image_path = os.path.join(plot_locations,
                                          str(model_id) + img_prefix + '_plot.png')
                # if(i ==0 and j ==0):
                #    image_db_path = image_short_path + get_only_file_name(csv_file_location) + img_prefix +  '_plot.png'
                sns.pairplot(data, x_vars=model_features[i], y_vars=model_labels[j], size=4, aspect=1, kind='scatter')
                plot_image = plot_image_path  # os.path.join(root_path, 'static/images/plots/', get_only_file_name(csv_file_location) + '_plot.png')
                plt.savefig(plot_image, dpi=300, bbox_inches='tight')
        shutil.make_archive(plot_zip_locations + str(model_id), 'zip', plot_locations)
        # plt.show()

        # ------------------Predict values from the model-------------------------#
        now = datetime.now()
        all_return_values = {'accuracy': c_m, 'confusion_matrix': c_m, 'plot_image_path': html_path,  # image_path,
                             'file_name': file_name,
                             'Mean_Absolute_Error': Mean_Absolute_Error,
                             'Mean_Squared_Error': Mean_Squared_Error,
                             'Root_Mean_Squared_Error': Root_Mean_Squared_Error,
                             'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                             'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

        # Add model profile to the database
        modelmodel = {'model_id': model_id,
                      'model_name': file_name,
                      'user_id': 1,
                      'model_headers': str(cvs_header)[1:-1],
                      'prediction_results_accuracy': str(c_m),
                      'mean_absolute_error': str(Mean_Absolute_Error),
                      'mean_squared_error': str(Mean_Squared_Error),
                      'root_mean_squared_error': str(Root_Mean_Squared_Error),
                      'plot_image_path': html_path,
                      'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                      'ds_source': ds_source,
                      'ds_goal': ds_goal}
        model_model = ModelProfile(**modelmodel)
        # Delete current profile
        model_model.query.filter().delete()
        db.session.commit()
        # Add new profile
        db.session.add(model_model)
        db.session.commit()

        # Add features, labels, and APIs details
        add_features_list = add_features(model_id, model_features)
        add_labels_list = add_labels(model_id, model_labels)
        api_details_id = random.randint(0, 22)
        api_details_list = add_api_details(model_id, api_details_id, 'v1')
        api_details_list = update_api_details_id(api_details_id)
        # db.session.commit()
        # db.session.expunge_all()
        # db.close_all_sessions

        # APIs details and create APIs document

        convert_data_to_sample(csv_file_location, 5)
        return all_return_values

    def run_classification_model(self, location_details, ds_goal, ds_source, is_local_data, featuresdvalues=['data'],
                                 classification_label=['category']):
        try:
            model_id = Helper.generate_model_id()
            initiate_model = BaseController.initiate_model(model_id)
            helper = Helper()

            # Prepare the date and creating the classifier model
            classificationcontrollerHelper = ClassificationControllerHelper()
            files_path = '%s%s%s%s' % (app_root_path, data_files_folder, model_id,
                                       '_files')  # this code need to be rephrase to find how to get local data for new model
            csv_file_path = '%s%s' % (df_location, session['fname'])
            file_extension = pathlib.Path(csv_file_path).suffix
            newfilename = os.path.join(df_location, str(model_id) + file_extension)
            os.rename(csv_file_path, newfilename)
            csv_file_path = newfilename
            file_name = get_only_file_name(csv_file_path)

            # Create datafile (data.txt)
            if (is_local_data == 'Yes'):
                folders_list = ControllersHelper.get_folder_structure(files_path, req_extensions=('.txt'))
                featuresdvalues = ['data']
                classification_label = ['category']
                data_set = classificationcontrollerHelper.create_classification_data_set(files_path, folders_list,
                                                                                         model_id)
            elif (is_local_data == 'csv'):
                data_set = classificationcontrollerHelper.create_classification_csv_data_set(csv_file_path, model_id)
            else:
                folders_list = helper.list_ftp_dirs(
                    location_details)  # classificationcontrollerHelper.get_folder_structure(files_path, req_extensions=('.txt'))
                data_set = classificationcontrollerHelper.create_FTP_data_set(location_details, folders_list, model_id)

            full_file_path = '%s%s%s%s%s%s' % (app_root_path, data_files_folder, model_id, '/', str(model_id), '.txt')
            docs = classificationcontrollerHelper.setup_docs(full_file_path)
            categories, most_common = classificationcontrollerHelper.print_frequency_dist(docs)
            # X_train, X_test, y_train, y_test = classificationcontrollerHelper.get_splits(docs)
            t_model = classificationcontrollerHelper.train_classifier(model_id, docs, categories)

            # Save model metadata
            # Add model profile to the database

            now = datetime.now()
            modelmodel = {'model_id': model_id,
                          'model_name': file_name,
                          'user_id': 1,
                          'train_precision': t_model['train_precision'],
                          'train_recall': t_model['train_recall'],
                          'train_f1': t_model['train_f1'],
                          'test_precision': t_model['test_precision'],
                          'test_recall': t_model['test_recall'],
                          'test_f1': t_model['test_f1'],
                          'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                          'ds_source': ds_source,
                          'ds_goal': ds_goal,
                          'status': config_parser.get('ModelStatus', 'ModelStatus.active'),
                          'deployed': config_parser.get('DeploymentStatus', 'DeploymentStatus.notdeployed'),
                          'description': 'No description added yet.'}

            model_model = ModelProfile(**modelmodel)
            db.session.commit()
            # Add new profile
            db.session.add(model_model)
            db.session.commit()

            # Add features, labels, and APIs details
            add_features_list = add_features(model_id, featuresdvalues)
            add_labels_list = add_labels(model_id, classification_label)
            api_details_id = random.randint(0, 22)
            api_details_list = add_api_details(model_id, api_details_id, 'v1')
            api_details_list = update_api_details_id(api_details_id)

            return_values = {
                'model_id': model_id,
                'model_name': file_name,
                'segment': 'createmodel',
                'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S"),
                'train_precision': t_model['train_precision'],
                'train_recall': t_model['train_recall'],
                'train_f1': t_model['train_f1'],
                'test_precision': t_model['test_precision'],
                'test_recall': t_model['test_recall'],
                'test_f1': t_model['test_f1'],
                'most_common': numpy.array(most_common),
                'categories': numpy.array(categories)
            }

            return return_values

        except  Exception as e:
            return e

    def categories_mapper(self, real_values, encoded_values):

        return 0

    def classify_text(self, test_text, model_name=''):
        classification_controller_helper = ClassificationControllerHelper()
        return classification_controller_helper.classify(test_text, model_name)

# b = run_demo_model1(root_path, 'diabetes.csv', ['Age'], '1', '2')
