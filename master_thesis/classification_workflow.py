from dataset import *
from representations import *
from classify import *


#######################################################################################################################
limit_num = 80
#######################################################################################################################

#   20Newsgroup
News_dataset = read20news_limited(limit_num) # get train & test set
#    d_set[0] --> everything for train set
#    d_set[1] --> everything for test set
#    d_set[*][0] --> labels of * set
#    d_set[*][1] --> texts of * set
train_text = News_dataset[0][1]    # store text of train set
train_labels = News_dataset[0][0]
test_text = News_dataset[1][1]    # store text of train set
test_labels = News_dataset[1][0]


#   make train, test representations
news_voc = voc_20newsgroup(train_text, limit_num)  # load of vocabulary to use it in BoW representations
train_txt_bow = BOW(news_voc, train_text)[0]
test_txt_bow = BOW(news_voc, test_text)[0]

#   classification
data_name = "Newsgroup"
sim_measure = "cosine"
train_vectors_dict = dataset_dictionary(train_labels, train_txt_bow, limit_num, data_name)[0]
test_label_predictions = averaged_representative_classification(train_vectors_dict, test_txt_bow, sim_measure)

#   evaluation
predicted_labels = []
for lista in test_label_predictions[1]:
    predicted_labels.append(lista[0])

accuracy = sklearn.metrics.accuracy_score(test_labels, predicted_labels)
print("accuracy score: "+accuracy)

F1 = sklearn.metrics.f1_score(test_labels, predicted_labels)
print("F1 score: "+F1)