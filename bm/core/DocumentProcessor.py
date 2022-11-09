#  Copyright (c) 2021. Slonos Labs. All rights Reserved.
import numpy
import numpy as np
import pandas as pd

from bm.datamanipulation.AdjustDataFrame import remove_null_values


class DocumentProcessor:
    custom_dtypes = []
    model_types = []

    def __init__(self):
        self.custom_dtypes = ['int64', 'float64', 'datetime', 'string']
        self.model_types = ['Prediction', 'Time Series Forecasting']

    def document_analyzer(self, csv_file_location):
        # Read the file
        df = pd.read_csv(csv_file_location)
        if (not df.empty) or (len(df.columns) < 2):
            total_rows = len(df.index)

            # list of columns data types
            columns_list = df.columns
            data_types = df.dtypes
            extracted_data_types = []
            datetime_columns = []
            numric_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                        datetime_columns.append(col)
                        extracted_data_types.append('datetime')
                    except ValueError:
                        extracted_data_types.append('string')
                        pass
                elif (df[col].dtype == 'float64' or df[col].dtype == 'int64'):
                    numric_columns.append(col)
                    extracted_data_types.append(df[col].dtype.name)
                else:
                    extracted_data_types.append('string')

            # Check if there is any empty columns
            df = df.replace(' ', np.nan)
            nan_cols = []
            for col in df.columns:
                x = pd.isna(df[col])
                x = x.to_numpy()
                if not False in x:
                    nan_cols.append(col)
            nan_cols = numpy.array(nan_cols)

            # Clean the data frame
            df = df.drop(columns=nan_cols, axis=1)
            final_columns_list = df.columns
            total_rows = len(df.index)
            df = remove_null_values(df)  # drop empty columns
            final_total_rows = len(df.index)  # Get number of rows after removing the null rows
            no_droped_coulmns = total_rows

            # According to data types, show suggessted models
            #print('We have reviwed the provided data and below is our reviewing results:')
            #print('1- You have data of (' + ', '.join(columns_list) + ')')
            #print('2- The columns (' + ', '.join(nan_cols) + ') are empty and will be removed before proccesing to create the model')
            #print('3- The final columns list to be used for creating the model is: [' + ', '.join(final_columns_list) + ']')
            #print('4- You have number of records that contains empty values in some columns, the model ignores any record that have empty values')
            #print('5- Number of rows after removing rows with some empty values is:' + str(final_total_rows) + '\n')
            #print('Based on above analysis results, BrontoMind can help you to do the following:')
            #print('1- Create prediction model to predict the value of one or more item from (' + ', '.join(columns_list) + ') and using the remaing columns as an input for this model.')
            #print('2- Build time series timeforecasting model to track the chages in the values of one from (' + ','.join(numric_columns)  + ') according the change in the date/time of one from: (' + ','.join(datetime_columns) + ')')
            return ', '.join(columns_list), ', '.join(nan_cols), ', '.join(final_columns_list), str(final_total_rows), ','.join(numric_columns), ','.join(datetime_columns)
        else:
            return -1
    def dataframe_summary(self,df):
        return 'We have reviwed the provided data and below is our reviewing results:' \
               '    1- You have  date of ' + 'columns_list'\
               '    2- The column/s' + 'nan_cols' +  'are empty and will be removed before proccesing to create the model' + \
               '    3- The final columns list is:' + 'final_columns_list' + \
               '    4- You have number of records that contains empty values in some columns' \
               '    5- Number of rows after removing rows with some empty values are:' + 'final_total_rows'

#d = DocumentProcessor()
#d.document_analyzer('covid_19_india.csv')
