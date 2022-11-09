# import csv
# import os
# import pickle
# import string
# from collections import defaultdict
# from os import listdir
# from os.path import isfile, join
# from random import shuffle
#
# from nltk import word_tokenize, FreqDist
# from nltk.corpus import stopwords
# from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
#
# root_path = os.getcwd()
#
# stop_words = set(stopwords.words('english'))
#
# def create_csv_data_file(output_csv_file_name, folder_location, header:['label', 'file_name', 'text'], req_extensions):
#     csv_folder_location = root_path + folder_location
#     csv_file_location = root_path + folder_location + '/' + output_csv_file_name
#     data = create_txt_data_file(csv_folder_location, req_extensions)
#     with open(csv_file_location, 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#
#         # write the header
#         writer.writerow(header)
#
#         # write the data
#         writer.writerows(data)
#
#         return 1
#
# def get_folder_structure(path_of_the_directory, req_extensions=('.txt')):
#     try:
#         full_path = path_of_the_directory
#         folders_list = [f for f in listdir(full_path) if isfile(join(full_path, f)) == False]  # get sub folders list
#         ext = req_extensions  # ex: ('txt', 'docx')
#         files_list = []
#         folder_structure = dict()
#
#         # for i in folders_list:
#         #     sub_folder_path = full_path + '/' + i
#         #     dictionary_fields = []
#         #     for file_name in os.listdir(sub_folder_path):
#         #         if file_name.endswith(ext):
#         #             dictionary_fields.append(file_name)
#         #         else:
#         #             continue
#         #     dic_keys = dictionary_fields
#         #     dic_values = dictionary_fields
#         #     folder_structure.update({i: dict(zip(dic_keys, dic_values))})
#         return folders_list #, folder_structure
#     except  Exception as e:
#         print(e)
#         return 0
#
# def create_txt_data_file(path_of_the_directory, req_extensions=('.txt')):
#     try:
#         full_path = path_of_the_directory
#         folders_list = [f for f in listdir(full_path) if isfile(join(full_path, f)) == False]  # get sub folders list
#         ext = req_extensions  # ex: ('txt', 'docx')
#         data_list = []
#
#         for i in folders_list:
#             sub_folder_path = full_path + '/' + i
#             dictionary_fields = []
#             for file_name in os.listdir(sub_folder_path):
#                 if file_name.endswith(ext):
#                     with open(sub_folder_path + '/' + file_name, 'rb') as file:
#                         file_text = file.readline().decode(errors='replace').replace('/n', '')
#                         data_list.append([i, file_name, file_text.strip()])
#                 else:
#                     continue
#         return data_list
#
#     except  Exception as e:
#         print(e)
#         return 0
#
# def print_frequency_dist(docs):
#     tokens = defaultdict(list)
#
#     for doc in docs:
#         doc_label = doc[0]
#         doc_text = doc[2]
#         doc_tokens = word_tokenize(doc_text)
#         tokens[doc_label].extend(doc_tokens)
#
#     for category_label, category_tokens in tokens.items():
#         print(category_label)
#         fd = FreqDist(category_tokens)
#         print(fd.most_common(20))
#
# def create_data_set(folder_path, labels):
#     try:
#         output_file = '%s%s' % (folder_path, 'data.txt')
#         with open(output_file, 'w', encoding='utf8') as outfile:
#             for label in labels:
#                 dir = '%s%s' % (folder_path, label)
#                 for filename in os.listdir(dir):
#                     fullfilename = '%s%s%s' % (dir, '/', filename)
#                     with open(fullfilename, 'rb') as file:
#                         text = file.read().decode(errors='replace').replace('\n', '')
#                         outfile.write('%s\t%s\t%s\n' % (label, filename, text))
#         return 1
#     except  Exception as e:
#         print(e)
#         return 0
#
# def setup_docs(full_file_path):
#     docs =[]
#
#     with open(full_file_path, 'r', encoding='utf8') as datafile:
#         for row in datafile:
#             parts = row.split('\t')
#             doc = (parts[0], parts[2].strip())
#
#             docs.append(doc)
#         return docs
#
# def get_tokens(text):
#     tokens = word_tokenize(text)
#     tokens = [t for t in tokens if not t in stopwords]
#     return tokens
#
# def clean_text(text):
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = text.lower()
#     return text
#
# def get_splits(docs):
#     shuffle(docs)
#
#     X_train= []
#     X_test= []
#     y_train= []
#     y_test= []
#
#     pivot = int(0.8 * len(docs))
#
#     for i in range(0, pivot):
#         X_train.append(docs[i][1])
#         y_train.append(docs[i][0])
#
#     for i in range(0, len(docs)):
#         X_test.append(docs[i][1])
#         y_test.append(docs[i][0])
#
#     return X_train, X_test, y_train, y_test
#
# def train_classifier(docs):
#     try:
#         X_train, X_test, y_train, y_test = get_splits(docs)
#
#         vectorized = CountVectorizer(stop_words='english', ngram_range=(1,3), min_df=3, analyzer='word')
#
#         # Create doc-term matrix
#         dtm = vectorized.fit_transform(X_train)
#
#         # Train the model
#         naive_bays_classifier = MultinomialNB().fit(dtm, y_train)
#
#         # Model evaluations
#         train_precision, train_recall, train_f1 = evaluate_classifier(naive_bays_classifier, vectorized, X_train, y_train)
#         test_precision, test_recall, test_f1 = evaluate_classifier(naive_bays_classifier, vectorized, X_test, y_test)
#
#
#         # store the classifier
#         clf_filename = root_path + '/class_pkl.pkl'
#         pickle.dump(naive_bays_classifier, open(clf_filename, 'wb'))
#
#         # Store vectorized
#         vic_filename = root_path + '/v_pkl.pkl'
#         pickle.dump(vectorized, open(vic_filename, 'wb'))
#
#         return  1
#     except  Exception as e:
#         print(e)
#         return 0
#
# def evaluate_classifier(classifier, vectorizer, X_test, y_test):
#     X_test_tfidf = vectorizer.transform(X_test)
#     y_pred = classifier.predict(X_test_tfidf)
#     precision =metrics.precision_score(y_test, y_pred,
#                                            pos_label='positive',
#                                            average='micro')
#     recall = metrics.recall_score(y_test, y_pred,
#                                            pos_label='positive',
#                                            average='micro')
#     f1 = metrics.f1_score(y_test, y_pred,
#                                            pos_label='positive',
#                                            average='micro')
#     print(("%s\t%s\t%s\n") % (precision, recall, f1))
#
#     return precision, recall, f1
#
# def classify(text):
#     # Load model
#     clf_filename = root_path + '/class_pkl.pkl'
#     np_clf = pickle.load(open(clf_filename,'rb'))
#
#     # load vectorizer
#     vec_filename = root_path + '/v_pkl.pkl'
#     vectorizer = pickle.load(open(vec_filename, 'rb'))
#
#     pred = np_clf.predict(vectorizer.transform([text]))
#
#     return pred
#
# #cc = setup_docs('/examples', ('.txt'))
#
# # files_path = root_path + '/examples/'
# # folders_list = get_folder_structure(files_path, req_extensions=('.txt'))
# # rr = create_data_set(files_path, folders_list)
# # full_file_path = root_path + '/examples/' + 'data.txt'
# # docs = setup_docs(full_file_path)
# # t_model = train_classifier(docs)
#
# cat = classify("Justice Department to review police response to Uvalde school shootingJustice Department to review police response to Uvalde school shooting")
# print(cat)
# print('Done')
