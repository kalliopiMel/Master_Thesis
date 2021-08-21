from classify import *
import numpy as np


def sorting_similarities(sim_dict):
    n = len(sim_dict)
    for key in sim_dict:
        sim_list = sim_dict[key]
        n = len(sim_list)
        # Traverse through all array elements
        for i in range(n - 1):
        # range(n) also work but outer loop will repeat one time more than needed.

         # Last i elements are already in place
            for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
                if sim_list[j][1] > sim_list[j + 1][1]:
                    sim_list[j], sim_list[j + 1] = sim_list[j + 1], sim_list[j]

    return sim_dict



def Sampling_texts_HE(texts, k, synt):
#    sim_text = text_pairs

# Gia kathe reference doc Pare akraies perioxes tou ranking RNK[:k], RNK[-k:] kai mesaia perioxh RNK[median-k/2:median:median+k/2]
    if synt == True:
        infile = open('ranked_syntax_sim','rb')
        ranked_dict = pickle.load(infile)
        infile.close()
    else:
        infile = open('ranked_sim','rb')
        ranked_dict = pickle.load(infile)
        infile.close()
# the text similarities
    eval_texts = []
    eval_dict = {}

    for key in ranked_dict:
        ranked_sim = ranked_dict[key]
        rs_length = len(ranked_sim)  # number of rows of the array  https://stackoverflow.com/questions/10713004/find-length-of-2d-array-python

        No_match = ranked_sim[:k]
        middle = rs_length//2
        mid_k = k//2 +1
        intermediate = ranked_sim[middle - mid_k: middle + mid_k]
        matching = ranked_sim[-k:]

        eval_dict[key] = []

        for x in No_match:
            cand = x[0]
            similar = x[1]
            eval_dict[key].append([cand, similar])
            eval_texts.append([texts[cand], similar])

        for x in intermediate:
            cand = x[0]
            similar = x[1]
            eval_dict[key].append([cand, similar])
            eval_texts.append([texts[cand], similar])

        for x in matching:
            cand = x[0]
            similar = x[1]
            eval_dict[key].append([cand, similar])
            eval_texts.append([texts[cand], similar])

    print("The selected texts for each reference in numbers")
    all_items = 0
    for key, value in eval_dict.items():
        all_items = all_items + len(eval_dict[key])
        print(key, ':', value)
    print("text items are ", all_items)
    return eval_texts


# Pare ta candidate docs apo autes tis perioxes, kai pare ta tuple permutations
# To reference R mazi me kathe tuple sta permutations auta ta dinoume gia human evaluation