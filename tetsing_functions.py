from classify import *
from dataset import *
from representations import *
limit_num = 10
#######################################################################################################################

#   20Newsgroup
News_dataset = read20news_limited(limit_num) # get train & test set
#    d_set[0] --> everything for train set
#    d_set[1] --> everything for test set
#    d_set[*][0] --> labels of * set
#    d_set[*][1] --> texts of * set
train_text = News_dataset[0][1]    # store text of train set
train_labels = News_dataset[0][0]
#test_text = News_dataset[1][1]    # store text of train set
#test_labels = News_dataset[1][0]


#   make train, test representations
news_voc = voc_20newsgroup(train_text, limit_num)  # load of vocabulary to use it in BoW representations
train_txt_bow = BOW(news_voc, train_text)
#test_txt_bow = BOW(news_voc, test_text)[0]
print("train_txt_bow")
print(train_txt_bow)
print("lexiko")
lexiko = Candidates_vs_Reference(train_txt_bow, "cosine")
print(lexiko)