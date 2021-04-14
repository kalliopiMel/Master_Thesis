import sklearn
# 3) build / define similarity extraction system for an input pair of texts

#   we build cosine similarity
#   in order to do that we import cosine similary library
#   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html?highlight=cosine%20similarity#sklearn.metrics.pairwise.cosine_similarity
#   https://datascience.stackexchange.com/questions/26648/cosine-similarity-returns-matrix-instead-of-single-value
from sklearn.metrics.pairwise import cosine_similarity

#   we calculate cosine similarity for every pair set existed in the arrays for train and test set
#   def cosine_similarity_func (text_array):
#       return sklearn.metrics.pairwise.cosine_similarity(text_array)

#print(texts_train_similarities)
