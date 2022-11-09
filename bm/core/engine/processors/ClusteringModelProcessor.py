#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
from sklearn.cluster import KMeans


class ClusteringModelProcessor:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def clusteringmodelselector(self, numberofrecords):
        try:
            numberofrecordsedge = 100000
            cls = KMeans(
                n_clusters=5, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0
            )
            return cls
        except Exception as e:
            print(e)
            return 0
