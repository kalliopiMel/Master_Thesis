#   we import 20newsgroup for the experiments
#   https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups

#   os.path is used to restore the already saved pickle files
#   https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
#   https://wiki.python.org/moin/UsingPickle
#   https://www.pitt.edu/~naraehan/python3/pickling.html
import os.path

# we import__(CountVectorizer) in order to create our vocabulary for the model to be trained from 20NewsGroup
# text train set
from sklearn.feature_extraction.text import CountVectorizer

#   pickle is used to store data in the computer
#   https://www.journaldev.com/15638/python-pickle-example
import pickle


#######################################################################################################################
#######################################################################################################################

def read20news_limited(limit=None):
    # 1) read 20Newsgroups dataset and load train and test set for a limited amount of data for program testing purposes
    # first i load labels(lbls) and texts(txts) for the train set and then i do the same for the test set
    if limit is None:
        train20_lbls = fetch_20newsgroups(subset='train').target
        train20_txts = fetch_20newsgroups(subset='train').data

        test20_lbls = fetch_20newsgroups(subset='test').target
        test20_txts = fetch_20newsgroups(subset='test').data
    else:
        train20_lbls = fetch_20newsgroups(subset='train').target[:limit]
        train20_txts = fetch_20newsgroups(subset='train').data[:limit]

        test20_lbls = fetch_20newsgroups(subset='test').target[:limit]
        test20_txts = fetch_20newsgroups(subset='test').data[:limit]

    #   return the discovered information
    return [train20_lbls, train20_txts], [test20_lbls, test20_txts]


###########################################################################################################
def voc_20newsgroup(texts_train, limit=None):
    #   i have to create my vocabulary for the corpus

    #   i first set the path of vocabulary's file  https://www.btelligent.com/en/blog/best-practice-working-with-paths-in-python-part-1/
    if limit is None:
        voc20_path = 'C:\\Users\kalli\PycharmProjects\master_thesis\\20News_Voc'
    else:
        voc20_path = 'C:\\Users\kalli\PycharmProjects\master_thesis\\20News_Voc_'+str(limit)
    #   and then i either create and save the vocabulary or load it.
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

    return vocabulary


def dataset_dictionary(labels, texts):
    #   create a dictionary with labels being the keys and the texts creating a list in front of every label representing
    #   the relative text
    text20_dict = {}  # dictionary initialisation
    txt_len = len(texts)  # length of dictionary
    # print(labels)
    for i in range(txt_len):  # creating the dictionary
        if labels[i] not in text20_dict.keys():
            text20_dict[labels[i]] = [texts[i]]
        else:
            text20_dict[labels[i]].append(texts[i])

    print(text20_dict.keys())
    #   store the dictionary  with pickle
    file = open('dictionary_of_TrainText', 'wb')  # create the file
    pickle.dump(text20_dict, file)
    file.close()

    return text20_dict


# for testing purposes we need the above function with limit so
def dataset_dictionary(labels, texts, dataset, limit=None):
    #   create a dictionary with labels being the keys and the texts creating a list in front of every label representing
    #   the relative text
    text20_dict = {}  # dictionary initialisation
    txt_len = len(texts)  # length of dictionary
    # print(labels)
    for i in range(txt_len):  # creating the dictionary
        if labels[i] in text20_dict.keys():
            text20_dict[labels[i]].append(texts[i])
        else:
            text20_dict[labels[i]] = [texts[i]]


    print(text20_dict.keys())
    #   store the dictionary  with pickle
    file = open('TrainText_dict_'+str(limit)+"_"+dataset, 'wb')  # create the file
    pickle.dump(text20_dict, file)
    file.close()

    return text20_dict


#function gia train and test set