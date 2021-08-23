#   we import 20newsgroup for the experiments
from sklearn.datasets import fetch_20newsgroups

#   os.path is used to restore the already saved pickle files
import os.path

# we import__(CountVectorizer) in order to create our vocabulary for the model to be trained from 20NewsGroup
# text train set
from sklearn.feature_extraction.text import CountVectorizer

#   pickle is used to store data in the computer
import pickle
import json
import random


def read20news():
    # 1) read 20Newsgroups dataset and load train and test set for a limited amount of data for program testing purposes
    # first i load labels(lbls) and texts(txts) for the train set and then i do the same for the test set
    cat_path = 'categories'
    if not os.path.exists(cat_path):
        cat20 = fetch_20newsgroups(subset='test').target_names
        cat = random.sample(cat20, 5)
        print(cat)
        with open("categories.json", "w") as f:
            json.dump(cat, f)
    with open("categories.json", "r") as f:
        categories = json.load(f)

    dataNews_path = 'Text_20News'
    dataNewsLab_path = 'Labels_20News'
    if not os.path.exists(dataNews_path):
        test_lbls_init = []
        test_txts_init = []
        for i in range(5):
            test_txts_temp = fetch_20newsgroups(subset='test', remove=(
                'headers', 'footers'), categories=[categories[i]]).data
            length = len(test_txts_temp)
            for j in range(5):
                r = random.randint(0, length-1)
                test_lbls_init.append(i)
                test_txts_init.append(test_txts_temp[r])
        f = open(dataNews_path, "wb")
        pickle.dump(test_txts_init, f)
        f.close()
        fl = open(dataNewsLab_path, "wb")
        pickle.dump(test_lbls_init, fl)
        fl.close()

    f = open(dataNews_path, "rb")
    test_txts = pickle.load(f)
    f.close()
    fl = open(dataNewsLab_path, "rb")
    test_lbls = pickle.load(fl)
    fl.close()

    #   return the discovered information
    return test_lbls, test_txts, categories


def voc_20newsgroup():
    #   i have to create my vocabulary for the corpus
    dataNews_Whole_path = 'Whole_20News'
    if not os.path.exists(dataNews_Whole_path):
        test_txts_temp = fetch_20newsgroups(
            subset='test', remove=('headers', 'footers')).data
        f = open(dataNews_Whole_path, "wb")
        pickle.dump(test_txts_temp, f)
        f.close()

    #   i first set the path of vocabulary's file  https://www.btelligent.com/en/blog/best-practice-working-with-paths-in-python-part-1/
    voc20_path = 'Whole_20News_Voc'
    #   and then i either create and save the vocabulary or load it.
    if not os.path.exists(voc20_path):
        f = open(dataNews_Whole_path, "rb")
        corpus = pickle.load(f)
        f.close()
        vocabulary = CountVectorizer().fit(corpus)  # we create 20NewsGroup vocabulary
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

###########################################################################################################


def voc_20newsgroup_limited():
    #   i have to create my vocabulary for the corpus
    dataNews_limited_path = 'Text_20News'
    #   i first set the path of vocabulary's file  https://www.btelligent.com/en/blog/best-practice-working-with-paths-in-python-part-1/
    voc20_limit_path = 'Limited_20News_Voc'
    #   and then i either create and save the vocabulary or load it.
    if not os.path.exists(voc20_limit_path):
        f = open(dataNews_limited_path, "rb")
        corpus_limited = pickle.load(f)
        f.close()
        # we create 20NewsGroup vocabulary
        vocabulary = CountVectorizer().fit(corpus_limited)
        #   we store the vocabulary
        file = open(voc20_limit_path, 'wb')  # we create a file to be written
        pickle.dump(vocabulary, file)  # we store to that file the vocabulary
        file.close()  # close the file
        # print(voc)
    else:
        file = open(voc20_limit_path, "rb")
        vocabulary = pickle.load(file)  # we store to that file the vocabulary
        file.close()

    return vocabulary


def MIcrosoft_Paraphrase():
    fname = 'msr_paraphrase_test.txt'
    with open(fname, encoding="utf8") as f:
        content = f.readlines()
    content.remove(content[0])


# function gia train and test set
