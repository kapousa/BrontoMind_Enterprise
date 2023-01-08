from flask import render_template, request

from bm.controllers.cvision.ObjectDetectionCotroller import ObjectDetectionCotroller


class ObjectDetectionDirector:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    @staticmethod
    def select_cvision_type():
        return render_template('applications/pages/cvision/selectvision.html', message='There is no active model')

    def createobjectdetection(self, ds_goal, ds_source):
        try:
            objectdetectingcotroller = ObjectDetectionCotroller()
            objectdetectionmodel = objectdetectingcotroller.create_model(ds_goal, ds_source)

            page_url =  "{0}cvision/{1}/objtdetect/detect".format(request.host_url, str(objectdetectionmodel['model_id']))
            page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
            return render_template('applications/pages/cvision/objectdetection/modelstatus.html',
                                   message='There is no active model',
                                   fname=objectdetectionmodel['model_name'],page_url=page_url, page_embed=page_embed,
                                   segment='createmodel', model_id=objectdetectionmodel['model_id'],
                                   created_on=objectdetectionmodel['created_on'],
                                   updated_on=objectdetectionmodel['updated_on'],
                                   last_run_time=objectdetectionmodel['last_run_time'],
                                   ds_goal=ds_goal, ds_sourc=ds_source
                                   )
        except Exception as e:
            print(e)
            return 0

    def showobjdetectrmodeldashboard(self,profile):
        page_url = "{0}cvision/{1}/objtdetect/detect".format(request.host_url, str(profile['model_id']))
        page_embed = "<iframe width='500' height='500' src='" + page_url + "'></iframe>"
        return render_template('applications/pages/cvision/objectdetection/dashboard.html',
                               message='No',
                               fname=profile['model_name'],page_url=page_url, page_embed=page_embed,
                               segment='showdashboard', created_on=profile['created_on'],
                               ds_goal=profile['ds_goal'],model_id=profile['model_id'],
                               updated_on=profile['updated_on'], last_run_time=profile['last_run_time'])

    def detect_object(self, model_id, runid, host, uname, pword):
        try:
            objectdetectioncontroller = ObjectDetectionCotroller()
            run_identifier = "%s%s%s" % (model_id, '_', runid)
            labelfileslink = objectdetectioncontroller.labelfiles(run_identifier, host, uname, pword)

            return render_template('applications/pages/cvision/objectdetection/labelfiles.html',
                                   message='No', labeled = 'Yes', model_id=model_id, run_id=runid,
                                   download= labelfileslink)
        except Exception as e:
            print(e)
            return 0



