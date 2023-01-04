from flask import render_template

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
            createthemodel = objectdetectingcotroller.create_model(ds_goal, ds_source)

            return render_template('applications/pages/cvision/objectdetection/modelstatus.html', message='There is no active model')
        except Exception as e:
            print(e)
            return 0
