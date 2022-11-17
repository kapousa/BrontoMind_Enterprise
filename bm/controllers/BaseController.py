import os
import shutil

from app import db
from app.base.constants.BM_CONSTANTS import scalars_location, pkls_location, output_docs_location, df_location, \
    plot_zip_locations, plot_locations, data_files_folder, pkls_files_folder, html_plots_location
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.base.db_models.ModelAPIModelMethods import ModelAPIModelMethods
from app.base.db_models.ModelEncodedColumns import ModelEncodedColumns
from app.base.db_models.ModelFeatures import ModelFeatures
from app.base.db_models.ModelForecastingResults import ModelForecastingResults
from app.base.db_models.ModelLabels import ModelLabels
from app.base.db_models.ModelProfile import ModelProfile
from bm.datamanipulation.AdjustDataFrame import deletemodelsfiles
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper import AttributesHelper
from bm.db_helper.AttributesHelper import get_model_name


class BaseController:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def deletemodels(self):
        try:
            ModelEncodedColumns.query.filter().delete()
            ModelFeatures.query.filter().delete()
            ModelLabels.query.filter().delete()
            ModelAPIModelMethods.query.filter().delete()
            ModelAPIDetails.query.filter().delete()
            ModelProfile.query.filter().delete()
            ModelForecastingResults.query.filter().delete()
            db.session.commit()

            # Delete old model files
            delete_model_files = deletemodelsfiles(scalars_location, pkls_location, output_docs_location, df_location, plot_zip_locations, plot_locations, data_files_folder, pkls_files_folder)

            return 1
        except Exception as e:
            print('Ohh -delet_models...Something went wrong.')
            print(e)
            return 0

    def delete_model(self, model_id=0):
        try:
            model_name = get_model_name(model_id)
            ModelEncodedColumns.query.filter_by(model_id = model_id).delete()
            ModelFeatures.query.filter_by(model_id = model_id).delete()
            ModelLabels.query.filter_by(model_id = model_id).delete()
            ModelAPIModelMethods.query.filter_by(model_id = model_id).delete()
            ModelAPIDetails.query.filter_by(model_id = model_id).delete()
            ModelProfile.query.filter_by(model_id = model_id).delete()
            ModelForecastingResults.query.filter_by(model_id = model_id).delete()
            db.session.commit()

            # Delete old model files
            delete_model_files = deletemodelsfiles(scalars_location, pkls_location, output_docs_location, df_location, plot_zip_locations, plot_locations, data_files_folder, pkls_files_folder)

            return 1
        except Exception as e:
            print('Ohh -delete_model...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def get_cm_accurcy(c_m):
        print(c_m)
        sum_of_all_values = 0
        number_of_columns = len(c_m[0])
        correct_pre_sum = 0

        for i in range(number_of_columns):
            correct_pre_sum = correct_pre_sum + c_m[i][i]

        c_m = c_m.flatten()
        for i in range(len(c_m)):
            sum_of_all_values = sum_of_all_values + c_m[i]

        c_m_accurcy = round((correct_pre_sum/sum_of_all_values),3) * 100

        return c_m_accurcy

    @staticmethod
    def get_all_models():
        try:
            model_profiles = ModelProfile.query.all()
            profiles = []

            for profile in model_profiles:
                model_profile = {'model_id': profile.model_id,
                                 'model_name': profile.model_name,
                                 'model_description': profile.description,
                                 'status': AttributesHelper.get_lookup_value(profile.status),
                                 'updated_on': profile.updated_on,
                                 'updated_by': AttributesHelper.get_user_fullname(profile.user_id),
                                 'ds_goal': profile.ds_goal,
                                 }
                profiles.append(model_profile)

            return profiles
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def get_model_status(model_id):
        try:
            model_profile_row = [ModelProfile.query.filter_by(model_id = model_id).first()]
            model_profile = {}

            for profile in model_profile_row:
                model_profile = {'model_id': profile.model_id,
                                 'model_name': profile.model_name,
                                 'prediction_results_accuracy': str(profile.prediction_results_accuracy),
                                 'mean_absolute_error': str(profile.mean_absolute_error),
                                 'mean_squared_error': str(profile.mean_squared_error),
                                 'root_mean_squared_error': str(profile.root_mean_squared_error),
                                 'plot_image_path': profile.plot_image_path,
                                 'created_on': profile.created_on,
                                 'updated_on': profile.updated_on,
                                 'last_run_time': profile.last_run_time,
                                 'ds_source': profile.ds_source,
                                 'ds_goal': profile.ds_goal,
                                 'mean_percentage_error': profile.mean_percentage_error,
                                 'mean_absolute_percentage_error': profile.mean_absolute_percentage_error,
                                 'depended_factor': profile.depended_factor,
                                 'forecasting_category': profile.forecasting_category,
                                 'train_precision': profile.train_precision,
                                 'train_recall': profile.test_f1,
                                 'train_f1': profile.test_f1,
                                 'test_precision': profile.test_f1,
                                 'test_recall': profile.test_f1,
                                 'test_f1': profile.test_f1,
                                 'description': profile.description,
                                 'status': profile.status
                                 }
                print(model_profile)
            return model_profile
        except  Exception as e:
            print('Ohh -get_model_status...Something went wrong.')
            print(e)
            return 0

    @staticmethod
    def initiate_model(model_name):

        ploting_path = html_plots_location + model_name
        zip_path = plot_zip_locations + model_name
        endodedcategories_path = DataCoderProcessor.category_location + model_name
        endodedcategoriespkls_path = DataCoderProcessor.pkls_location + model_name


        # Delet old folder
        shutil.rmtree(ploting_path) if (os.path.isdir(ploting_path)) else print(0)
        shutil.rmtree(zip_path) if (os.path.isdir(zip_path)) else print(0)
        shutil.rmtree(endodedcategories_path) if (os.path.isdir(endodedcategories_path)) else print(0)
        shutil.rmtree(endodedcategoriespkls_path) if (os.path.isdir(endodedcategoriespkls_path)) else print(0)

        # Create new folder
        os.mkdir(ploting_path)
        os.mkdir(zip_path)
        os.mkdir(endodedcategories_path)
        os.mkdir(endodedcategoriespkls_path)

        return 0