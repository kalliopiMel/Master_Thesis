import sklearn

#    BOW for test & train,  (create function for BOW)
def BOW (vocab, texts):
    #   we transform texts to vector using the 20NewsGroup voc for both test and train set
    texts_array = vocab.transform(texts).toarray()
#    print(voc.transform(traintxt))
#    print(texts_train_array)
    return texts_array