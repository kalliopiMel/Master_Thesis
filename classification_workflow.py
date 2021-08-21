from dataset import *
from representations import *
from classify import *
from human_evaluation import *
import pickle
import numpy as np
from syntactic_parser import *

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
k = 2
syntax = False
label_predictions = Candidates_vs_Reference(News_txt_bow, sim_measure)
file = open('reference_dict', 'wb')  # create the file
pickle.dump(label_predictions[0], file)
file.close()

f = open('similarities_list', 'wb')  # create the
# file
pickle.dump(label_predictions[1], f)
f.close()
print("How many times ref and candidate were found more similar")
for key,value in label_predictions[0].items():
	print(key, ':', value)


ranked_results = sorting_similarities(label_predictions[0])
f = open('ranked_similarities', 'wb')  # create the file
pickle.dump(ranked_results, f)
f.close()
print("those similarities ranked")
for key,value in ranked_results.items():
	print(key, ':', value)


text_for_evaluation = Sampling_texts_HE(News_text, k, syntax)
print("having the text for evaluation")
f = open("Human_Evaluation.txt", "wb")  #  write results in .txt file
pickle.dump(text_for_evaluation[0], f)
f.close()
i = 0
for x in text_for_evaluation[0]:
	i+=1
	print(x)
print("Number of selected texts when k=", k, ": ", i)
#the dictionary
f = open("Human_Eval_dictionary.txt", "wb")  #  write results in .txt file
pickle.dump(text_for_evaluation[1], f)
f.close()


print("Same work for syntactic test")
# and the SYNTAX part!
syntax_BOW_text = Text_syntaxBOW(News_text, News_txt_bow)
syntax = True
print("How many times ref and candidate were found more similar")
label_syntax_predictions = Candidates_vs_Reference(syntax_BOW_text, sim_measure)
file = open('reference_synt_dict', 'wb')  # create the file
pickle.dump(label_syntax_predictions[0], file)
file.close()

f = open('similarities_synt_list', 'wb')  # create the
# file
pickle.dump(label_syntax_predictions[1], f)
f.close()
for key,value in label_syntax_predictions[0].items():
	print(key, ':', value)



print("those similarities ranked")
ranked_syntax_results = sorting_similarities(label_syntax_predictions[0])
f = open('ranked_syntax_sim', 'wb')  # create the file
pickle.dump(ranked_syntax_results, f)
f.close()
for key,value in ranked_syntax_results.items():
	print(key, ':', value)


print("having the text for evaluation")
text_for_syntax_evaluation = Sampling_texts_HE(News_text, k, syntax)

print("Number of selected texts when k=", k, ": ", i)
f = open("Human_Syntax_Evaluation.txt", "wb")  #  write results in .txt file
pickle.dump(text_for_syntax_evaluation[0], f)
f.close()
i = 0
for x in text_for_syntax_evaluation[0]:
	i+=1
	print(x)
f = open("Human_Syntax_Eval_dictionary.txt", "wb")  #  write results in .txt file
pickle.dump(text_for_syntax_evaluation[1], f)
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
