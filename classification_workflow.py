import pdb
from dataset import *
from representations import *
from classify import *
from human_evaluation import *
import pickle
import numpy as np
from syntactic_parser import *

#   20Newsgroup
# News_dataset = read20news() # get train & test set
News_labels, News_text, categories = read20news()  # store text of train set

f = open("News_text", "wb")
pickle.dump(News_text, f)
f.close()

#   make train, test representations
# load of vocabulary to use it in BoW representations
news_voc = voc_20newsgroup_limited()
News_txt_bow = BOW(news_voc, News_text)

#   something like classification
data_name = "Newsgroup"
sim_measure = "cosine"
k = 2
syntax = False
label_predictions = Candidates_vs_Reference(News_txt_bow, sim_measure)
pw_sims = label_predictions[2]
file = open('reference_dict', 'wb')  # create the file
pickle.dump(label_predictions[0], file)
file.close()

f = open('similarities_list', 'wb')  # create the
# file
pickle.dump(label_predictions[1], f)
f.close()
print("How many times ref and candidate were found more similar")
for key, value in label_predictions[0].items():
    print(key, ':', value)


ranked_results = sorting_similarities(label_predictions[0])
f = open('ranked_similarities', 'wb')  # create the file
pickle.dump(ranked_results, f)
f.close()
print("those similarities ranked")
for key, value in ranked_results.items():
    print(key, ':', value)


text_for_evaluation = Sampling_texts_HE(News_text, k, syntax)
print("having the text for evaluation")
f = open("Human_Evaluation.txt", "wb")  # write results in .txt file
pickle.dump(text_for_evaluation[0], f)
f.close()
i = 0
for x in text_for_evaluation[0]:
    i += 1
    print(x)
print("Number of selected texts when k=", k, ": ", i)
# the dictionary
f = open("Human_Eval_dictionary", "wb")  # write results in .txt file
pickle.dump(text_for_evaluation[1], f)
f.close()


###########################################################################################################
##################################			SYNTAX				###########################################
###########################################################################################################

# and the SYNTAX part!

print("###########################################################################################################")
print("###########################################################################################################")
print("Same work for syntactic test")
syntax_BOW_text = Text_syntaxBOW(News_text, News_txt_bow)
syntax = True
print("How many times ref and candidate were found more similar")
label_syntax_predictions = Candidates_vs_Reference(
    syntax_BOW_text, sim_measure)
pw_sims_syntactic = label_syntax_predictions[2]
file = open('reference_synt_dict', 'wb')  # create the file
pickle.dump(label_syntax_predictions[0], file)
file.close()

f = open('similarities_synt_list', 'wb')  # create the
# file
pickle.dump(label_syntax_predictions[1], f)
f.close()
for key, value in label_syntax_predictions[0].items():
    print(key, ':', value)


print("those similarities ranked")
ranked_syntax_results = sorting_similarities(label_syntax_predictions[0])
f = open('ranked_syntax_sim', 'wb')  # create the file
pickle.dump(ranked_syntax_results, f)
f.close()
for key, value in ranked_syntax_results.items():
    print(key, ':', value)


print("having the text for evaluation")
text_for_syntax_evaluation = Sampling_texts_HE(News_text, k, syntax)

print("Number of selected texts when k=", k, ": ", i)
f = open("Human_Syntax_Evaluation.txt", "wb")  # write results in .txt file
pickle.dump(text_for_syntax_evaluation[0], f)
f.close()
i = 0
for x in text_for_syntax_evaluation[0]:
    i += 1
    print(x)

f = open("Human_Syntax_Eval_dictionary", "wb")  # write results in .txt file
pickle.dump(text_for_syntax_evaluation[1], f)
f.close()


def make_human_eval_inputs(prefix, heval, text_data):
    with open(prefix + "_" + "heval_inputs_required_evaluations.json", "w") as f:
        # keep only ids
        heval = {key: [x[0] for x in v] for (key, v) in heval.items()}
        heval = {key: [str(x) for x in v] for (key, v) in heval.items()}
        for key in heval:
            vals = heval[key]
            midpt = int(
                len(vals)/2 - 1) if len(vals) % 2 == 0 else int(len(vals)/2)
            halfk = int(k/2)
            newvals = vals[midpt: midpt + halfk + 1]
            assert len(newvals) == k, "Wrong middle slice of doc rankings"
            newvals = vals[:k] + newvals + vals[-k:]
            heval[key] = newvals
        json.dump(heval, f)

    with open(prefix + "_" + "heval_inputs_texts_ids.json", "w") as f:
        txts = {str(i): txt for (i, txt) in enumerate(text_data)}
        json.dump(txts, f)


make_human_eval_inputs("syntax", text_for_syntax_evaluation[1], News_text)
make_human_eval_inputs("no_syntax", text_for_evaluation[1], News_text)

#   evaluation
# predicted_labels = []
# for lista in test_label_predictions[1]: #needs to be corrected
# 	predicted_labels.append(lista[0])

# accuracy = sklearn.metrics.accuracy_score(News_labels, predicted_labels)
# print("accuracy score: "+accuracy)

# F1 = sklearn.metrics.f1_score(News_labels, predicted_labels)
# print("F1 score: "+F1)


#   for the human evaluation
