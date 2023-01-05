import numpy
import pandas as pd
from flask import render_template, session

from app.base.constants.BM_CONSTANTS import progress_icon_path, loading_icon_path, root_path, df_location, demo_key, \
    docs_templates_folder, output_docs
from app.base.db_models.ModelAPIDetails import ModelAPIDetails
from app.base.db_models.ModelProfile import ModelProfile
from bm.apis.v1.APIHelper import APIHelper
from bm.controllers.prediction.ModelController import ModelController
from bm.datamanipulation.DataCoderProcessor import DataCoderProcessor
from bm.db_helper.AttributesHelper import get_labels, get_features, get_model_name
from bm.utiles.CVSReader import improve_data_file


class PredictionDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']


    def fetch_data(self, fname, headersArray, message):
        return render_template('applications/pages/prediction/selectfields.html', headersArray=headersArray,
                               fname=fname,
                               ds_source=session['ds_source'], ds_goal=session['ds_goal'],
                               segment='createmodel', message=message)

    def creatingthemodel(self, request, fname, ds_goal, ds_source):
        predictionvalues = numpy.array((request.form.getlist('predcitedvalues')))
        featuresdvalues = numpy.array((request.form.getlist('featuresdvalues')))

        return render_template('applications/pages/prediction/creatingpredictionmodel.html',
                               predictionvalues=predictionvalues,
                               featuresdvalues=featuresdvalues,
                               progress_icon_path=progress_icon_path, fname=fname,
                               loading_icon_path=loading_icon_path,
                               ds_source=ds_source, ds_goal=ds_goal,
                               segment='createmodel')

    def complete_the_model(self, request):
        fname = request.form.get('fname')
        ds_source = request.form.get('ds_source')
        ds_goal = request.form.get('ds_goal')
        data_file_path = "%s%s" % (df_location, fname)
        df = pd.read_csv(data_file_path, sep=",")
        data_sample = (df.sample(n=5))
        predictionvalues = numpy.array((request.form.getlist('predcitedvalues')))
        featuresdvalues = numpy.array((request.form.getlist('featuresdvalues')))
        idf = improve_data_file(fname, df_location, predictionvalues)

        # run model
        modelcontroller = ModelController()
        model_controller = modelcontroller.run_prediction_model(root_path, data_file_path, featuresdvalues, predictionvalues, ds_source,
                                                ds_goal, demo_key)

        if model_controller == 0:
            return render_template('page-501.html',
                                   error="There is no enugh data to build the model after removing empty rows. The data set should have mimimum 50 records to buld the model.",
                                   segment='error')

        # Webpage details
        page_url = request.host_url + "predictevalues?t=" + str(ds_goal) + "&s=" + str(ds_source) + "&m=" + str(model_controller['model_id'])
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"

        # APIs details and create APIs document
        model_api_details = ModelAPIDetails.query.first()
        apihelper = APIHelper()
        model_head = ModelProfile.query.with_entities(ModelProfile.model_id, ModelProfile.model_name).filter_by(
            model_id=model_controller['model_id']).first()
        generate_apis_docs = apihelper.generateapisdocs(0, model_head.model_id,
                                                        str(request.host_url + 'api/' + model_api_details.api_version),
                                                        docs_templates_folder, output_docs)

        return render_template('applications/pages/prediction/modelstatus.html',
                               accuracy=model_controller['accuracy'],
                               confusion_matrix=model_controller['confusion_matrix'],
                               plot_image_path=model_controller['plot_image_path'], sample_data=[
                data_sample.to_html(border=0, classes='table table-hover', header="false",
                                    justify="center").replace(
                    "<th>", "<th class='text-warning'>")],
                               Mean_Absolute_Error=model_controller['Mean_Absolute_Error'],
                               Mean_Squared_Error=model_controller['Mean_Squared_Error'],
                               Root_Mean_Squared_Error=model_controller['Root_Mean_Squared_Error'],
                               segment='createmodel', page_url=page_url, page_embed=page_embed,
                               created_on=model_controller['created_on'],
                               updated_on=model_controller['updated_on'],
                               last_run_time=model_controller['last_run_time'],
                               fname=model_controller['file_name'])

    def predict_labels(self, request):
        try:
            ds_goal = request.args.get("t")
            ds_source = request.args.get("s")
            model_id = request.args.get("m")
            features_list = get_features(model_id)
            labels_list = get_labels(model_id)
            testing_values = []
            opt_param = len(request.form)
            all_gategories_values = DataCoderProcessor.get_all_categories_values(model_id)

            if opt_param == 0:
                # response = make_response()
                return render_template('applications/pages/prediction/predictevalues.html', features_list=features_list,
                                       labels_list=labels_list, ds_goal=ds_goal,
                                       predicted_value='nothing', testing_values='nothing',
                                       all_gategories_values=all_gategories_values, predicted='Nothing', message='No')
            else:
                if request.method == 'POST':
                    for i in features_list:
                        feature_value = request.form.get(i)
                        # final_feature_value = float(feature_value) if feature_value.isnumeric() else feature_value
                        final_feature_value = feature_value
                        testing_values.append(final_feature_value)
                    modelcontroller = ModelController()
                    predicted_value = modelcontroller.predict_values_from_model(model_id, testing_values)
                    # response = make_response()
                    return render_template('applications/pages/prediction/predictevalues.html',
                                           features_list=features_list,
                                           labels_list=labels_list, ds_goal=ds_goal,
                                           predicted_value=predicted_value, testing_values=testing_values,
                                           all_gategories_values=all_gategories_values, predicted='Yes', message='No')
        except Exception as e:
            return render_template('applications/pages/nomodeltopredictevalues.html',
                                   error=str(e),
                                   message="Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset",
                                   segment='message')
