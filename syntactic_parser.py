from sklearn.feature_extraction.text import CountVectorizer
import os.path
#  syntax parser of spacy
import spacy
from representations import *
import numpy as np
import pickle
#nlp = spacy.load("en_core_web_sm")
# for label in nlp.get_pipe("parser").labels: # that way i can find the different labels of dependency parser


def Syntax_Text(text_list):
    nlp = spacy.load("en_core_web_sm")
    syntax = []
    for text in text_list:
        doc = nlp(text)
        temp = []
        for token in doc:
            temp.append(token.pos_)
        syntax.append(" ".join(temp))
    return syntax


def Text_syntaxBOW(text, text_BOW):

    syntactic_text = Syntax_Text(text)

    # vocab
    if os.path.exists("syntax_vocab"):
        with open("syntax_vocab", "rb") as f:
            syntactic_cv = pickle.load(f)
    else:
        syntactic_cv = CountVectorizer()
        syntactic_cv.fit(syntactic_text)
        with open("syntax_vocab", "wb") as f:
            syntactic_cv = pickle.dump(syntactic_cv, f)

    syntactic_vectors = syntactic_cv.transform(syntactic_text).todense()
    with open("syntax_vectors", "wb") as f:
        pickle.dump(syntactic_vectors, f)

    concatenated = np.concatenate((text_BOW, syntactic_vectors), axis=1)
    concatenated = concatenated.tolist()
    return concatenated
