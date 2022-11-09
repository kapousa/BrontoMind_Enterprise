class BMEngine:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    # Used for transfer and kind of data to csv file format before creating the model
    def prepare_dateset(self, dataset, datasource, datagoal):

        return 0

    def datasource_mapping(self, datasource):
        switcher = {
            0: "zero",
            1: "one",
            2: "two",
        }
        return switcher.get(datasource, "nothing")
