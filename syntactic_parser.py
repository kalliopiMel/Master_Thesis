# all information from   https://www.districtdatalabs.com/syntax-parsing-with-corenlp-and-nltk

#from nltk.parse.corenlp import CoreNLPServer
#import os   #  info for os https://www.pythonforbeginners.com/os/pythons-os-module  https://docs.python.org/3/library/os.html

#  syntax parser of spacy https://spacy.io/usage/linguistic-features#pos-tagging
#  purpose of different labels
# had to follow the installation steps from here https://stackoverflow.com/questions/66367475/oserror-e050-cant-find-model-en-core-web-sm-it-doesnt-seem-to-be-a-short
import spacy
nlp = spacy.load("en_core_web_sm")
for label in nlp.get_pipe("parser").labels: # that way i can find the different labels of dependency parser https://stackoverflow.com/questions/58215855/how-to-get-full-list-of-pos-tag-and-dep-in-spacy
   # labels can also be found here https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    print(label, " -- ", spacy.explain(label))
doc = nlp("The red Apple is looking at buying U.K.'s first ever startup for 1$ billion. The company intents to show off their power buy anouncing fase advertisement. Thus, the company is pressured by Apple to give in.")

# model "en_core_web_sm" couldn't be found so i used python -m spacy download en_core_web_sm command spa on pycharm terminal to download it and thus use it

for token in doc:
   print(token.text, token.pos_, token.dep_)





# The server needs to know the location of the following files:
#   - stanford-corenlp-X.X.X.jar
#   - stanford-corenlp-X.X.X-models.jar
#STANFORD = os.path.join("models", "stanford-corenlp-full-2018-02-27")

# Create the server
#server = CoreNLPServer(
#   os.path.join(STANFORD, "stanford-corenlp-3.9.1.jar"),
#   os.path.join(STANFORD, "stanford-corenlp-3.9.1-models.jar"),
#)

# Start the server in the background
#server.start()