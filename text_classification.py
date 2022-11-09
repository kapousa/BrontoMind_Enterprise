# from sklearn.datasets import fetch_20newsgroups
# from pprint import pprint
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics
# import numpy as np
#
#
# newsgroups_train = fetch_20newsgroups(subset='train')
# # print(newsgroups_train.keys())
# # pprint(list(newsgroups_train.target_names))
# newsgroups_train.filenames.shape
# newsgroups_train.target.shape
# newsgroups_train.target[:10]
# cats = ['alt.atheism', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
# list(newsgroups_train.target_names)
# newsgroups_train.filenames.shape
# newsgroups_train.target.shape
# newsgroups_train.target[:10]
# categories = ['alt.atheism', 'talk.religion.misc',
#               'comp.graphics', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train',
#                                       categories=categories)
#
#
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(newsgroups_train.data)
# vectors.shape
# vectors.nnz / float(vectors.shape[0])
# newsgroups_test = fetch_20newsgroups(subset='test',
#                         categories=categories)
#
# vectors_test = vectorizer.transform(newsgroups_test.data)
#
# clf = MultinomialNB(alpha=.01)
# clf.fit(vectors, newsgroups_train.target)
# pred = clf.predict(vectors_test)
# metrics.f1_score(newsgroups_test.target, pred, average='macro')
#
# def show_top10(classifier, vectorizer, categories):
#     feature_names = np.asarray(vectorizer.get_feature_names())
#     for i, category in enumerate(categories):
#         top10 = np.argsort(classifier.coef_[i])[-10:]
#         # print("%s: %s" % (category, " ".join(feature_names[top10])))
#
# show_top10(clf, vectorizer, newsgroups_train.target_names)
# newsgroups_test = fetch_20newsgroups(subset='test',
#                                       remove=('headers', 'footers', 'quotes'),
#                                       categories=categories)
# vectors_test = vectorizer.transform(newsgroups_test.data)
# pred = clf.predict(vectors_test)
# metrics.f1_score(pred, newsgroups_test.target, average='macro')
#
#
# newsgroups_train = fetch_20newsgroups(subset='train',
#                                        remove=('headers', 'footers', 'quotes'),
#                                        categories=categories)
# vectors = vectorizer.fit_transform(newsgroups_train.data)
# clf = MultinomialNB(alpha=.01)
# clf.fit(vectors, newsgroups_train.target)
# vectors_test = vectorizer.transform(newsgroups_test.data)
# pred = clf.predict(vectors_test)
# print(pred)
# metrics.f1_score(newsgroups_test.target, pred, average='macro')
