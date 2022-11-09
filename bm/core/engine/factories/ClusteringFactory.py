from app.base.app_routes.directors.ClassificationDirector import ClassificationDirector
from app.base.app_routes.directors.ClusteringDirector import ClusteringDirector
from bm.core.engine.BMModelFactory import BMModelFactory


class ClusteringFactory(BMModelFactory):

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    #----------------- Classification models -------------------#
    def create_clustering_csv_model(self, request):
        return 0

    def create_clustering_db_model(self, request):
        return 0

    def create_classification_gs_model(self, request):
        return 0

    def create_clustering_sf_model(self, request):
        return 0

    def create_clustering_text_model(self, request):
        clusterung_director = ClusteringDirector()
        return clusterung_director.create_text_clustering_model(request)
    # ----------------- End Classification models -------------------#




