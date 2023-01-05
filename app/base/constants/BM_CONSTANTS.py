#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import os

from flask import request

pkls_location = 'pkls/'

scalars_location = 'scalars/'

plot_locations = 'app/static/images/plots/'

plot_zip_locations = 'app/base/static/zip/'

plot_zip_download_location = 'base/static/zip/'

image_short_path = 'static/images/plots/'

df_location = 'app/data/'

clusters_keywords_file = 'app/data/clusters_keywords.pkl'

number_of_clustering_keywords = 25

html_plots_location = 'app/templates/applications/plots/'

html_short_path = 'applications/plots/'

image_location = 'app/'

root_path = '../app/'

app_root_path = 'app/'

clustering_root_path = 'app/'

output_docs_location = 'app/base/output_docs/'

docs_templates_folder = 'docs_templates/'

output_docs = 'app/base/output_docs/'

data_files_folder = 'data/'

pkls_files_folder = 'app/pkls'

physical_allowed_extensions = ['txt']

loading_icon_path = os.path.join('images/', 'loading_icon.gif')

progress_icon_path = os.path.join('images/', 'progress_icon_2.gif')

labeled_data_filename_download_path = 'base/output_docs/'

labeled_data_filename = 'data.csv'

demo_key = 'DEMO'

output_document_sfx = '_BrontoMind_APIs_document.docx'

prediction_model_keyword = 'Prediction'

classification_model_keyword = 'Classification'

clustering_model_keyword = 'Clustering'

forecasting_model_keyword = 'Forecasting'

dep_path = 'deployed/'

scripts_path = '/scripts/'
