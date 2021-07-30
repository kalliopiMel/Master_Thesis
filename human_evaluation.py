from classify import *
import numpy as np


def Ranking_Similaries_HE(ranked_sim):
    # i have to transform my ranked_similarities table into an 2d array, so that iτ can βε sortεδ and useεδ for human evaluation task
    ranked_sim = np.array(ranked_sim)
    print(ranked_similarities)
    columnIndex = 2
    # Sort 2D numpy array by 2nd Column
    ranked_sim = ranked_sim[ranked_sim[:, columnIndex].argsort()]
    print('Sorted 2D Numpy Array')
    print(ranked_sim)

    # ranked_sim now contains the ref text, the cand text and the amount of times they were found similar

    f = open('ranked_sim', 'wb')  # create the file
    pickle.dump(ranked_sim, f)
    f.close()

    return ranked_sim


def Sampling_texts_HE(texts, k):
    sim_text = text_pairs

# Gia kathe reference doc Pare akraies perioxes tou ranking RNK[:k], RNK[-k:] kai mesaia perioxh RNK[median-k/2:median:median+k/2]
    infile = open('ranked_sim','rb')
    ranked_sim = pickle.load(infile)
    infile.close()

    rs_length = np.shape(ranked_sim)[0]    # number of rows of the array  https://stackoverflow.com/questions/10713004/find-length-of-2d-array-python

    No_match = ranked_sim[k, 2*k]
    intermediate = ranked_sim[rs_length//2, rs_length//2 + k]
    matching = ranked_sim[rs_length - 2*k, rs_length - k]

    # the text similarities
    eval_texts = []
    No_match_l = No_match.tolist()              #transformning array to list  https://datatofish.com/numpy-array-to-list-python/
    intermediate_l = intermediate.tolist()
    matching_l = matching.tolist()

    for x in No_match_l:
        x[0] = ref
        x[1] = cand
        x[2] = similar
        eval_texts.append([texts[ref], texts[cand], similar])

    for x in intermediate_l:
        x[0] = ref
        x[1] = cand
        x[2] = similar
        eval_texts.append([texts[ref], texts[cand], similar])

    for x in matching_l:
        x[0] = ref
        x[1] = cand
        x[2] = similar
        eval_texts.append([texts[ref], texts[cand], similar])

    return eval_texts


# Pare ta candidate docs apo autes tis perioxes, kai pare ta tuple permutations
# To reference R mazi me kathe tuple sta permutations auta ta dinoume gia human evaluation