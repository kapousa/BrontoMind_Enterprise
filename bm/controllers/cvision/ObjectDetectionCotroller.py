import random
import subprocess

from flask import session
from sqlalchemy.sql.functions import now

from app import config_parser, db
from app.base.constants.BM_CONSTANTS import scripts_path
from app.base.db_models.ModelCvisionRun import ModelCvisionRun
from app.base.db_models.ModelProfile import ModelProfile
from bm.controllers.BaseController import BaseController
from bm.db_helper.AttributesHelper import add_api_details, update_api_details_id
from bm.utiles.Helper import Helper
from datetime import datetime


class ObjectDetectionCotroller:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def train_model(self):
        return 0

    def create_model(self, ds_goal, ds_source):
        return self._create_object_detecting_model(ds_goal, ds_source)

    def _create_object_detecting_model(self, ds_goal, ds_source):
        try:
            model_id = Helper.generate_model_id()
            model_name = 'CVision_{}'.format(model_id)
            now = datetime.now()
            initiate_model = BaseController.initiate_model(model_id)

            all_return_values = {'model_id': model_id,
                                 'model_name': model_name,
                                 'created_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'updated_on': now.strftime("%d/%m/%Y %H:%M:%S"),
                                 'last_run_time': now.strftime("%d/%m/%Y %H:%M:%S")}

            # Add model profile to the database
            modelmodel = {'model_id': model_id,
                          'model_name': model_name,
                          'user_id': session['logger'],
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
            api_details_id = random.randint(0, 22)
            api_details_list = add_api_details(model_id, api_details_id, 'v1')
            api_details_list = update_api_details_id(api_details_id)

            return all_return_values

        except  Exception as e:
            base_controller = BaseController()
            base_controller.deletemodel(model_id)
            print(e)
            return -1

    def labelfiles(self, run_identifier, desc, host, uname, pword):
        return self._lable_files(run_identifier, desc, host, uname, pword)

    def _lable_files(self, run_identifier, desc, host, uname, pword): # Run identifier is combination from model_id_run_id
        try:
            # Label files
            uploadtargetfiles = self._upload_target_files(run_identifier, host, uname, pword)
            detect_script_location = "%s%s" % (scripts_path, 'runyolo.sh')
            subprocess.call(['sh', '.' + detect_script_location, 'data/images', run_identifier])

            # Archiving results
            archive_script_location = "%s%s" % (scripts_path, 'archiveresults.sh')
            subprocess.call(['sh', '.' + archive_script_location, run_identifier])

            # Add run record to db
            runids = run_identifier.split('_')
            modelmodel = {
                "model_id":runids[0],
                "run_id": runids[1],
                "run_on": datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                "description": desc
            }
            model_model = ModelCvisionRun(**modelmodel)
            db.session.add(model_model)
            db.session.commit()

            return 1
        except Exception as e:
            return -1

    def _upload_target_files(self, model_id, host, uname, pword):
        try:
            script_location = "%s%s" % (scripts_path, 'downloadfiles.sh')
            subprocess.call(['sh', '.' + script_location, host, uname, pword, '/temp', model_id])

            return 1
        except Exception as e:
            return -1

