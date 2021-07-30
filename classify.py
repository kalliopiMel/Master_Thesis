from numpy import dot
from numpy.linalg import norm   # https://www.statology.org/cosine-similarity-python/
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def averaged_representative_classification(train_dict_vec, test_vectors, sim_measure):
    average_dict_vec = {}
    for i in train_dict_vec:  # label i in dictionary    ######### print(train_dict_vec[i])
        # average = average_vector(train_dict_vec[i])
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


#   comparisons of reference text with 2 candidate texts and then keep the ranking of how many wins, each text had over the others
def Candidates_vs_Reference(text_list, measure):
    reference_rank = {}  # the dictionary of every reference's list of similarities
    ranked_similarities = []
    candidate_rank = []  # in this list we count how many times a candidate text was the most similar one
    text_list_length = len(text_list)
    for i in range(text_list_length):
        reference_rank[i] = []  # initalise reference_rank dictionary
        candidate_rank.append(0)  # initialise list

    pairwise_sim = "R \t \t C1 \t \t sim1 \t \t C2 \t \t sim2 \n"
    f = open("pairwise_sim_triplet.txt",
             "w")  # starting writing a .txt file to save the similarity results and relations
    f.write("R \t \t C1 \t \t sim1 \t \t C2 \t \t sim2 \n")

    for i in range(text_list_length):  # every text in the text list is a reference text.
        ref = text_list[i]

        for j in range(text_list_length - 1):  # and for every reference text i want to check the proximity with all the rest
                                              # the last text on the list has already been compare with all the other from the point of k variant bellow
            cand1 = text_list[j]  # candidate text 1
            if j != i:   # but i obviously have to leave out the reference text, since candidate 1 cannot also be the reference text
                for k in range(j + 1, text_list_length):  # and every candidate 1 has to be compared with all the rest
                    cand2 = text_list[k]  # candidate text 2
                    if k != i:   # but again, not the reference text obviously
                        if measure == "cosine":  # to measure proximity we use cosine
                            sim1 = dot(ref, cand1)/(norm(ref)*norm(cand1))
                            sim2 = dot(ref, cand2)/(norm(ref)*norm(cand2))
                    #pairwise_sim = pairwise_sim + i, " \t \t ", j, " \t \t ", sim1, "\t \t ", k, "\t \t", sim2, "\n"  # we also write everything down on a string or file
                            f.write(str(i) + " \t \t " + str(j) + " \t \t " + str(sim1) + "\t \t " + str(k) + "\t \t" + str(sim2) + "\n")
                        if sim1 >= sim2:  # if sim1>=sim2 then the cand1 is closer to the reference than cand2, thus we give it one point, otherwise the point goes to cand2
                            candidate_rank[j] = candidate_rank[j] + 1  # what we count here is how many times each text was the most proximate compared to anothter
                        else:
                            candidate_rank[k] = candidate_rank[k] + 1

        for j in range(text_list_length):
            if j != i:  # again we don't need the reference text
                reference_rank[i].append([j, candidate_rank[j]])    # creating the dictionary, where key is the ref text. Every key has a relative list, containing the other texts and how many times where found more similar than other
                ranked_similarities.append([i,j,candidate_rank[j]]) # this creates a 2D array of this form [[ref, cand*, sim*],[ref,cand*,sim*],....] where cand* is the most similar of the two candidates and the relative similarity number sim*


        for j in range(text_list_length):  # initialise list again
            candidate_rank[j] = 0


    f.close()  # closing the file


    file = open('reference_rank', 'wb')  # create the file
    pickle.dump(reference_rank, file)
    file.close()

    f = open('ranked_similarities', 'wb')  # create the file
    pickle.dump(ranked_similarities, f)
    f.close()

    return reference_rank, ranked_similarities
