from dataset import *
from representations import *
from classify import *
from human_evaluation import *
import spacy
import pickle

#   20Newsgroup
#News_dataset = read20news() # get train & test set
News_text = read20news()[1]  # store text of train set
News_labels = read20news()[0]


#   make train, test representations
news_voc = voc_20newsgroup_limited()  # load of vocabulary to use it in BoW representations
News_txt_bow = BOW(news_voc, News_text)

#   something like classification
data_name = "Newsgroup"
sim_measure = "cosine"
test_label_predictions = Candidates_vs_Reference(News_txt_bow, sim_measure)
print("How many times ref and candidate were found more similar")
print(test_label_predictions)
ranked_results = Ranking_Similaries_HE(test_label_predictions[1])
print("those similarities ranked")
print(ranked_results)
text_for_evaluation = Sampling_texts_HE(News_text, 3)
print("having the text for evaluation")
print(text_for_evaluation)
f = open("Human_Evaluation.txt", "wb")  #  write results in .txt file
pickle.dump(text_for_evaluation, f)
f.close()




#   evaluation
predicted_labels = []
for lista in test_label_predictions[1]: #needs to be corrected
    predicted_labels.append(lista[0])

accuracy = sklearn.metrics.accuracy_score(News_labels, predicted_labels)
print("accuracy score: "+accuracy)

F1 = sklearn.metrics.f1_score(News_labels, predicted_labels)
print("F1 score: "+F1)


#   for the human evaluation
