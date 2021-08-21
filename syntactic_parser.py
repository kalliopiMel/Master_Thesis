from sklearn.feature_extraction.text import CountVectorizer
import os.path
#  syntax parser of spacy
import spacy
from representations import *
import numpy as np
import pickle
#nlp = spacy.load("en_core_web_sm")
#for label in nlp.get_pipe("parser").labels: # that way i can find the different labels of dependency parser


def Syntax_Voc():
    syntax_words = []
    nlp = spacy.load("en_core_web_sm")
    voc_path = 'syntax_vocab'
    #   and then i either create and save the vocabulary or load it.
    if not os.path.exists(voc_path):    # we create  vocabulary from the different syntactic labels of the parser
        for label in nlp.get_pipe("parser").labels:
            syntax_words.append(label)
        syntax_voc = CountVectorizer().fit(syntax_words)
        print(syntax_voc)

        #   we store the vocabulary
        file = open(voc_path, 'wb')  # we create a file to be written
        pickle.dump(syntax_voc, file)  # we store to that file the vocabulary
        file.close()  # close the file
        # print(voc)
    else:
        file = open(voc_path, "rb")
        syntax_voc = pickle.load(file)  # we store to that file the vocabulary
        file.close()
    return syntax_voc

def Syntax_Text(text_list):
    nlp = spacy.load("en_core_web_sm")
    syntax = []
    for text in text_list:
        doc = nlp(text)
        temp = []
        for token in doc:
            temp.append(token.pos_)
        string = ""
        for i in temp:
            string = string + i
        syntax.append(string)
    return syntax


def Text_syntaxBOW(text, text_BOW):
    vocab = Syntax_Voc()
    syntactic_text = Syntax_Text(text)
    text_syntaxBOW = BOW(vocab, syntactic_text)
    text_syntaxBOW = np.array(text_syntaxBOW)
    text_BOW = np.array(text_BOW)
    temp = np.concatenate((text_BOW, text_syntaxBOW), axis = 1)
    new_txtBOW = temp.tolist()
    return new_txtBOW
