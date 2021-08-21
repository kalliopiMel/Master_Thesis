from numpy import dot
from numpy.linalg import norm   # https://www.statology.org/cosine-similarity-python/
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
import os.path

#   comparisons of reference text with 2 candidate texts and then keep the ranking of how many wins, each text had over the others

def Reference_Text():
    ref_path = 'C:\\Users\kalli\PycharmProjects\master_thesis\\News_Ref'
    if not os.path.exists(ref_path):
        ref_rank = {}  # the dictionary of every reference's list of similarities
        for i in range(0, 25, 5):
            r1 = random.randint(i, i+4)
            ref_rank[r1]=[]
            r2 = random.randint(i, i+4)
            while r1 == r2:
                r2 = random.randint(i, i + 4)
            ref_rank[r2]=[]

        f = open(ref_path, "wb")
        pickle.dump(ref_rank, f)
        f.close()
    else:
        f = open(ref_path, "rb")
        ref_rank = pickle.load(f)
        f.close()

    return ref_rank


def Candidates_vs_Reference(text_list, measure):
    reference_rank = Reference_Text()
    ranked_similarities = []
    candidate_rank = []  # in this list we count how many times a candidate text was the most similar one
    text_list_length = len(text_list)
    for i in range(0,25):
        candidate_rank.append(0)  # initialise list

    pairwise_sim = []

    for i in reference_rank:  # every text in the text list is a reference text.
        ref = text_list[i]

        for j in range(text_list_length):  # and for every reference text i want to check the proximity with all the rest
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
                            pairwise_sim.append([i, j, sim1, k, sim2])
                        if sim1 >= sim2:  # if sim1>=sim2 then the cand1 is closer to the reference than cand2, thus we give it one point, otherwise the point goes to cand2
                            candidate_rank[j] = candidate_rank[j] + 1  # what we count here is how many times each text was the most proximate compared to anothter
                        else:
                            candidate_rank[k] = candidate_rank[k] + 1

        for j in range(text_list_length):
            if j != i:  # again we don't need the reference text
                reference_rank[i].append([j, candidate_rank[j]])    # creating the dictionary, where key is the ref text. Every key has a relative list, containing the other texts and how many times where found more similar than other
                ranked_similarities.append([i,j,candidate_rank[j]]) # this creates a 2D array of this form [[ref, cand*, sim*],[ref,cand*,sim*],....] where cand* is the most similar of the two candidates and the relative similarity number sim*


        for j in range(25):  # initialise list again
            candidate_rank[j] = 0

    # starting writing a .txt file to save the similarity results and relations
    f = open("Reference_Candidate_Times.txt", "w")
    f.write("R \t \t Cand  \t \t Times \n")
    for i in reference_rank:
        for j in reference_rank[i]:
            f.write(str(i) + " \t \t " + str(j[0]) + " \t \t " + str(j[1]) + "\n")
    f.close()  # closing the file


    file = open('reference_rank', 'wb')  # create the file
    pickle.dump(reference_rank, file)
    file.close()

    f = open('ranked_similarities', 'wb')  # create the file
    pickle.dump(ranked_similarities, f)
    f.close()

    return reference_rank, ranked_similarities


