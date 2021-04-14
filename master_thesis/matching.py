import sklearn
from representations import *


########################################################################################################################
########################################################################################################################
# Συνάρτηση σύγκρισης τριάδων κειμένων με σκοπό την εύρεση της πιο όμοιας 2-άδας
#  Δυάδες κειμένων και σύγκριση ομοιότητάς τους από λίστα με αποτελέσματα αντίστοιχων συγκρίσεων
# δημιουργία τέτοιας λίστας μέσω του 20newsgroup
# Task2: make a function with inputs: two texts and similarity extraction parameters (e.g. vocabulary)
# outputs: a similarity value
def two_out_of_three(voc, txt1, txt2):
    #   transform texts to vectors
    t1_vec = voc.transform([txt1]).toarray()
    print(t1_vec)
    t2_vec = voc.transform([txt2]).toarray()
    #   check cosine similarity
    return sklearn.metrics.pairwise.cosine_similarity(t1_vec, t2_vec)


########################################################################################################################
########################################################################################################################


#   what we want to do here is a function having as input three texts and as output their two pairs,
#   a matching pair and a no-match pair

def Two_out_of_3_Match(voc, maintxt, cand1, cand2):
#   first thing here is to create our vectors using the BOW function
    maintxt_v = BOW(voc, maintxt)
    cand1_v = BOW(voc, cand1)
    cand2_v = BOW(voc, cand2)
#   then we compute the cosine similarity of each pair
#   the way we conduct the comparisons is: two candidates are being compared with a main text sending both similarity
#   scores to the program
    sim1 = sklearn.metrics.pairwise.cosine_similarity(maintxt_v, cand1_v)
    sim2 = sklearn.metrics.pairwise.cosine_similarity(maintxt_v, cand2_v)
    return sim1, sim2


"""def Two_out_of_4_Match(voc, maintxt, cand1, cand2, cand3):
#   first thing here is to create our vectors using the BOW function
    maintxt_v = BOW(voc, maintxt)
    cand1_v = BOW(voc, cand1)
    cand2_v = BOW(voc, cand2)
    cand3_v = BOW(voc, cand3)
#   then we compute the cosine similarity of each pair
#   the way we conduct the comparisons is: two candidates are being compared with a main text sending both similarity
#   scores to the program
    sim1 = sklearn.metrics.pairwise.cosine_similarity(maintxt_v, cand1_v)
    sim2 = sklearn.metrics.pairwise.cosine_similarity(maintxt_v, cand2_v)
    sim3 = sklearn.metrics.pairwise.cosine_similarity(maintxt_v, cand3_v)
    return sim1, sim2, sim3"""
