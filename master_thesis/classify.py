#from maths import *
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy


def averaged_representative_classification(train_dict_vec, test_vectors, sim_measure):
    average_dict_vec = {}
    for i in train_dict_vec:    # label i in dictionary    ######### print(train_dict_vec[i])
        #average = average_vector(train_dict_vec[i])
        average_dict_vec[i] = numpy.mean(train_dict_vec[i])
    test_dict_vec = {}
    predictions = []
    for test_vec in test_vectors:
        sim = [0, -1]
        for label in train_dict_vec:
            label_vector = train_dict_vec[label]
            if sim_measure == "cosine":
                temp_sim = sklearn.metrics.pairwise.cosine_similarity(label_vector, test_vec)
            else:
                print("ERROR: Undefined measure")
                return None
            if temp_sim > sim[1]:
                sim = [label, temp_sim]
        if sim[0] in test_dict_vec:
            test_dict_vec[sim[0]].append(test_vec)
        else:
            test_dict_vec[sim[0]] = [test_vec]
        predictions.append(sim)
    return test_dict_vec, predictions

