from dataset import *
from representations import *
from classify import *
import numpy

d_set = read20news_limited(2) # get train & test set
texts_train = fetch_20newsgroups(subset='train').data[:8]
labels_train = fetch_20newsgroups(subset='train').target[:8]
news_voc = voc_20newsgroup(texts_train)
vector_data = BOW(news_voc, texts_train)
#print(texts_train)
#print(news_voc.get_feature_names()) function get_features_names():  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#print(vector_data)
d_set = read20news_limited(2) # get train & test set
#print(d_set[0][0])  # labels of tain set

text20_tr_dict = {}  # dictionary initialisation
txt_tr_len = len(texts_train)  # length of dictionary
# print(labels_train)
for i in range(txt_tr_len):  # creating the dictionary
    if labels_train[i] not in text20_tr_dict.keys():
        text20_tr_dict[labels_train[i]] = [texts_train[i]]
    else:
        text20_tr_dict[labels_train[i]].append(texts_train[i])

#print(labels_train)
#print("llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")

#averaged_representative_classification(text20_tr_dict, 1, 0)
v = [[1,2,3], [2,0,4], [1,1,3]]
#from maths import average_vector
#print(average_vector(v))
from numpy import * # https://thispointer.com/sorting-2d-numpy-array-by-column-or-row-in-python/
import numpy as np
v = np.array(v)
columnIndex = 1
# Sort 2D numpy array by 2nd Column
sortedArr = v[v[:,columnIndex].argsort()]
print('Sorted 2D Numpy Array')
print(sortedArr)