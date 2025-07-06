#pip install gensim
#Topic modelling - ml technique used in text analysis to discover underlying topics in a collection of documents
#vector space - algebraic model for repesenting text documents as vectors 


from gensim.models import keyedvectors     #used to load pre-trained word vectors
import gensim.downloader as api            #used to download pre-trained models from the Gensim data repository

#Load pre-trained Word2Vec model- tained by twitter with 25 dimensional word vector
word2vec_model = api.load("glove-twitter-25")

#Load pre-trained GloVe model - trained on Wikipedia and Gigaword with 100 dimensional word vector
glove_model = api.load("glove-wiki-gigaword-100")

#Word2Vec similar words
print("word2vec - words similar to 'king':")
print(word2vec_model.most_similar("king", topn =5))

#word2vec: word Analogy
print("\n word2vec - Analogy(king -man + woman):")
result = word2vec_model.most_similar(positive = ['king', 'woman'], negative = ['man'], topn=1)
print(result)


#GloVe similar words
print("\n Glove - words similar to 'king' :" )
print(glove_model.most_similar("king", topn=5))

#Glove: word analogy
print("\nGlove - Anlaogy (king - man + woman):")
result = glove_model.most_similar(positive =['king', 'woman'], negative =['man'], topn=1 )
print(result)



#output: {cosine similarity}

#word2vec - words similar to 'king':
#[('prince', 0.9337409734725952), ('queen', 0.9202421307563782), ('aka', 0.9176921844482422), ('lady', 0.9163240790367126), ('jack', 0.9147354364395142)]

#word2vec - Analogy(king -man + woman):
#[('meets', 0.8841924071311951)]             xxxx {bad result bcoz the dataset is small}

#Glove - words similar to 'king' :
#[('prince', 0.7682328820228577), ('queen', 0.7507690787315369), ('son', 0.7020888328552246), ('brother', 0.6985775232315063), ('monarch', 0.6977890729904175)]

#Glove - Anlaogy (king - man + woman):
#[('queen', 0.7698540687561035)]