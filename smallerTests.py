from dataset import *
fname = r'C:\Users\kalli\PycharmProjects\master_thesis\msr_paraphrase_test.txt'
with open(fname, encoding="utf8") as f:
    content = f.readlines()
print(content)
content.remove(content[0])
print(content)

k = read20news()
print(k)
v = voc_20newsgroup_limited()
voc = voc_20newsgroup()
