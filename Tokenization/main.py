# pip install nltk spacy
# python -m spacy download en_core_web_sm

# Import libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download both punkt and punkt space
nltk.download('')

# Sample text
text = "I love NLP! It's fun and exciting."


# NLTK: Word Tokenization
print("NLTK Word Tokenization:")
words = word_tokenize(text)
print(words)

# NLTK: Sentence Tokenization
print("\nNLTK Sentence Tokenization:")
sentences = sent_tokenize(text)
print(sentences)

# spaCy: Word and Sentence Tokenization
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
print("\nspaCy Word Tokenization:")
print([token.text for token in doc])
print("\nspaCy Sentence Tokenization:")
print([sent.text for sent in doc.sents])