from flask import session

from app.base.app_routes.directors.cvision.ObjectDetectionDirector import ObjectDetectionDirector


class ObjectDetectionFactory:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def selectcvisiontype(self, request):
        session['ds_goal'] = request.args.get("t")
        return ObjectDetectionDirector.select_cvision_type()

    def createobjectdetectionmodel(self, request):
        session['ds_soure'] = request.form.get('ds_source')
        objectdetectiondirector = ObjectDetectionDirector()

        return objectdetectiondirector.createobjectdetection(session['ds_goal'], session['ds_soure'])

    def showobjdetectrmodeldashboard(self, profile):
        objectdetectiondirector = ObjectDetectionDirector()

        return objectdetectiondirector.showobjdetectrmodeldashboard(profile)