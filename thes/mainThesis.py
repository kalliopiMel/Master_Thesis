#   i use sklearn since it provides a great deal of packages and libraries as far as it concerns NLP
import sklearn
#   we import 20newsgroup for the experiments
#   https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
import numpy
from numpy import array
#   Represantations is a file including all the function used for the text to be represented in python e.g. BOW
from Representations import *
# we import__(CountVectorizer) in order to create our vocabulary for the model to be trained from 20NewsGroup
# text train set
from sklearn.feature_extraction.text import CountVectorizer
#   pickle is used to store data in the computer
#   https://www.journaldev.com/15638/python-pickle-example
import pickle
#   os.path is used to restore the already saved pickle files
#   https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
#   https://wiki.python.org/moin/UsingPickle
#   https://www.pitt.edu/~naraehan/python3/pickling.html
import os.path
#   find the similarities of the above representations
#   includes all the functions used for similarity computation
from sklearn.metrics.pairwise import cosine_similarity
#   includes all the function that 
from About_Matching import *


#####################################################################################################################
#####################################################################################################################
#   check sci-kit version
print(sklearn.__version__)
# print("getting dataset!")
# x=12
#######################################################################################################################
#######################################################################################################################


# 1) read dataset  20Newsgroups dataset
#    read, train, test
#   for prototype reasons we limit data to  100
news20_labels = fetch_20newsgroups(subset='train').target[:100]  # target contains the labels of a document
news20_texts = fetch_20newsgroups(subset='train').data[:100]  # data contains the documents and texts

#   in this step we separate texts of train and tests set and similarly for labels
texts_train = news20_texts[:80]
texts_test = news20_texts[80:]
labels_train = news20_labels[:80]
labels_test = news20_labels[80:]

#######################################################################################################################
#######################################################################################################################
# 2) build representation for each text
#    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?fbclid=IwAR0ZF0LJqXJe_7QuZDTsEYwnqCfG-hRO8cZte3mjZL7KlG-laIQDWGbTC3Q


voc20_path = 'C:\\Users\\kalli\\PycharmProjects\\thes\\20News_Voc'
if not os.path.exists(voc20_path):
    vocabulary = CountVectorizer().fit(texts_train)  # we create 20NewsGroup vocabulary
    #   we store the vocabulary
    file = open(voc20_path, 'wb')  # we create a file to be written
    pickle.dump(vocabulary, file)  # we store to that file the vocabulary
    file.close()  # close the file
    # print(voc)
else:
    file = open(voc20_path, "rb")
    vocabulary = pickle.load(file)  # we store to that file the vocabulary
    file.close()

    #    create BOW representation for test & train



bow_repr_train = BOW(vocabulary, texts_train)
bow_repr_test = BOW(vocabulary, texts_test)


train_cosSim = sklearn.metrics.pairwise.cosine_similarity(bow_repr_train)
test_cosSim = sklearn.metrics.pairwise.cosine_similarity(bow_repr_test)

########################################################################################################################
########################################################################################################################
# 4) Evaluate the extracted similarities automatically
#   convert 20newsgroups labelset to pairs, save result with pickle
#   https://www.w3schools.com/python/python_dictionaries_nested.asp
#   create a dictionary with labels being the keys and the texts creating a list in front of every label representing
#   the relative text
text20_tr_dict = {}  # dictionary initialisation
txt_tr_len = len(texts_train)  # length of dictionary
# print(labels_train)
for i in range(txt_tr_len):  # creating the dictionary
    if labels_train[i] not in text20_tr_dict.keys():
        text20_tr_dict[labels_train[i]] = [texts_train[i]]
    else:
        text20_tr_dict[labels_train[i]].append(texts_train[i])

print(text20_tr_dict.keys())
#   store the dictionary  with pickle
file = open('dictionary_of_TrainText', 'wb')  # create the file
pickle.dump(text20_tr_dict, file)
file.close()

#   we use itertools and combinations to report all the possible non-ordered text pairs of texts that match
from itertools import combinations

#   20newsgroup list of matching pairs
for i in text20_tr_dict:
    dict_pairs_list = list(combinations(text20_tr_dict[i], 2))

file = open('dict_TrainText_Pairs', 'wb')  # save the list
pickle.dump(dict_pairs_list, file)
file.close()

########################################################################################################################
########################################################################################################################
# Συνάρτηση σύγκρισης τριάδων κειμένων με σκοπό την εύρεση της πιο όμοιας 2-άδας
#  Δυάδες κειμένων και σύγκριση ομοιότητάς τους από λίστα με αποτελέσματα αντίστοιχων συγκρίσεων
# δημιουργία τέτοιας λίστας μέσω του 20newsgroup


# print(texts_train[2],texts_train[37])
print(two_out_of_three(vocabulary, texts_train[2], texts_train[37]))

########################################################################################################################
########################################################################################################################
length = len(texts_train)  # it is easier to just use a variable than computing the length everytime

#   Moving on we create to list, one icluding the matchung texts and another for those that don't match
Matching_list = []
NoMatch_list = []
print("length", length)

#    now it's time to run the similarity function for all the possible pairs give from the corpus
for t in range(length):
    for t_ena in range(t + 1, length):
        for t_duo in range(t_ena + 1, length):
            #        print(t)
            #        print(t_ena)
            #        print(t_duo)
            sim = Two_out_of_3_Match(vocabulary, [texts_train[t]], [texts_train[t_ena]], [texts_train[t_duo]])
            #   i use sim_txt* in order to reduce the reading load of the big elements of the list
            sim_txt1 = [t, t_ena, sim[0][0][0]]
            sim_txt2 = [t, t_duo, sim[1][0][0]]

            #        print(sim[0][0][0], sim[1][0][0])
            #   if the similarity of the first pair is greater than the second then the first is included in the Matching_list
            #   and the other one in the NoMatch_list
            #   and vice versa
            if sim[0][0][0] >= sim[1][0][0]:
                #   i check before i include a similaraty match in the relative list, is that element is alreafy there so
                #   that i avoid doubles-
                if sim_txt1 not in Matching_list and sim_txt1 not in NoMatch_list:
                    Matching_list.append(sim_txt1)
                if sim_txt2 not in NoMatch_list and sim_txt2 not in Matching_list:
                    NoMatch_list.append(sim_txt2)
            else:
                if sim_txt1 not in NoMatch_list and sim_txt1 not in Matching_list:
                    NoMatch_list.append(sim_txt1)
                if sim_txt2 not in Matching_list and sim_txt2 not in NoMatch_list:
                    Matching_list.append(sim_txt2)

"""InBetweenPairs = []
for t in range(length):
    for t_ena in range(t + 1, length):
        for t_duo in range(t_ena + 1, length):
            for t_tria in range(t_duo + 1, length):
                #        print(t)
                #        print(t_ena)
                #        print(t_duo)
                sim = Two_out_of_4_Match(vocabulary, [texts_train[t]], [texts_train[t_ena]], [texts_train[t_duo]], [texts_train[t_tria]])
                #   i use sim_txt* in order to reduce the reading load of the big elements of the list
                #   i do the same with similar* so that i don't carry sim[][][] around
                similar1 = sim[0][0][0]
                similar2 = sim[1][0][0]
                similar3 = sim[2][0][0]
                sim_txt1 = [t, t_ena, similar1]
                sim_txt2 = [t, t_duo, similar2]
                sim_txt3 = [t, t_tria, similar3]

                #   out of the three pairs one is the max and one the min and the third one falls inbetween
                #   with that being said, max similarity pair goes in the Matching_list
                #   min similarity pair goes in the NoMatch_list
                #   there is a third list now the InBetweenPairs where we list the pairs that don't much with the 2 above

                if similar1 >= similar2 and similar1 >= similar3:
                    #   i check before i include a similaraty match in the relative list, is that element is alreafy there
                    #   so that i avoid doubles
                    if sim_txt1 not in Matching_list and sim_txt1 not in InBetweenPairs:
                        Matching_list.append(sim_txt1)
                    #   in case that a pair was in InBetweenPairs list we have to remove it now, because it falls on
                    #   specific category and we have to do that every time a pair is in 
                    #if sim_txt1 in InBetweenPairs:
                    #    InBetweenPairs.remove(sim_txt1)
                    if similar2 >= similar3:
                        if sim_txt2 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt2)
                        if sim_txt3 not in NoMatch_list:
                            NoMatch_list.append(sim_txt3)
                    else:
                        if sim_txt3 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt3)
                        if sim_txt2 not in NoMatch_list:
                            NoMatch_list.append(sim_txt2)
                elif similar2 >= similar1 and similar2 >= similar3:
                    #   i check before i include a similaraty match in the relative list, is that element is alreafy there
                    #   so that i avoid doubles
                    if sim_txt2 not in Matching_list:
                        Matching_list.append(sim_txt2)
                    if similar1 >= similar3:
                        if sim_txt1 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt1)
                        if sim_txt3 not in NoMatch_list:
                            NoMatch_list.append(sim_txt3)
                    else:
                        if sim_txt3 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt3)
                        if sim_txt1 not in NoMatch_list:
                            NoMatch_list.append(sim_txt1)
                elif similar3 >= similar2 and similar3 >= similar1:
                    #   i check before i include a similaraty match in the relative list, is that element is alreafy there
                    #   so that i avoid doubles
                    if sim_txt3 not in Matching_list:
                        Matching_list.append(sim_txt3)
                    if similar2 >= similar1:
                        if sim_txt2 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt2)
                        if sim_txt1 not in NoMatch_list:
                            NoMatch_list.append(sim_txt1)
                    else:
                        if sim_txt1 not in InBetweenPairs:
                            InBetweenPairs.append(sim_txt1)
                        if sim_txt2 not in NoMatch_list:
                            NoMatch_list.append(sim_txt2)"""
#print(Matching_list)
#print(NoMatch_list)
koina = 0
NoMatch_length = len(NoMatch_list)
Matching_length = len(Matching_list)
#InBetween_length = len(InBetweenPairs)
if NoMatch_length >= Matching_length:
    for i in NoMatch_list:
        if i in Matching_list:
            koina = koina + 1
else:
    for i in Matching_list:
        if i in NoMatch_list:
            koina = koina + 1
print("κοινά αρχεία", koina)
print("No Match list length", NoMatch_length, "percentage of common files in the two lists", koina/NoMatch_length)
print("Matching list length", Matching_length, "percentage of common files in the two lists", koina/Matching_length)

#   check the min and max vaues in the list
max_match = 0
max_nomatch = 0
min_match = 1
min_nomatch = 1
for i in NoMatch_list:
    if i[2] > max_nomatch:
        max_nomatch = i[2]
    if i[2] < min_nomatch:
        min_nomatch = i[2]
for i in Matching_list:
    if i[2] > max_match:
        max_match = i[2]
    if i[2] < min_match:
        min_match = i[2]


#print("κοινά αρχεία", koina)
#print("No Match list length", NoMatch_length, "percentage of common files in the two lists", koina/NoMatch_length)
print("Max value in NoMatch list is: ", max_nomatch, "and the min value is: ", min_nomatch)
#print("Matching list length", Matching_length, "percentage of common files in the two lists", koina/Matching_length)
print("Max value in Matching list is: ", max_match, "and the min value is: ", min_match)
