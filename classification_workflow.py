from dataset import *
from representations import *
from classify import *
from human_evaluation import *


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
train_txt_bow = BOW(news_voc, train_text)
test_txt_bow = BOW(news_voc, test_text)

#   something like classification
data_name = "Newsgroup"
sim_measure = "cosine"
train_vectors_dict = dataset_dictionary(train_labels, train_txt_bow,  data_name, limit_num)[0]
#test_label_predictions = averaged_representative_classification(train_vectors_dict, test_txt_bow, sim_measure)
test_label_predictions_2 = Candidates_vs_Reference(test_txt_bow, sim_measure)
print("How many times ref and candidate were found more similar")
print(test_label_predictions_2)
ranked_results = Ranking_Similaries_HE(test_label_predictions_2[1])
print("those similarities ranked")
print(ranked_results)
text_for_evaluation = Sampling_texts_HE(test_text, 3)
print("having the text for evaluation")
print(text_for_evaluation)


#   evaluation
predicted_labels = []
for lista in test_label_predictions_2[1]: #needs to be corrected
    predicted_labels.append(lista[0])

accuracy = sklearn.metrics.accuracy_score(test_labels, predicted_labels)
print("accuracy score: "+accuracy)

F1 = sklearn.metrics.f1_score(test_labels, predicted_labels)
print("F1 score: "+F1)


#   for the human evaluation
