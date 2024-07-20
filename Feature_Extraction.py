"""
Count Vectorizer and TF-IDF
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# load the data
file_path = 'D:/Desktop/study in France/ESIGELEC-study/Intership/IPSOS/cleaned_data_for_model.xlsx'
data = pd.read_excel(file_path)

# Extracting cleaned_text columns from a DataFrame
documents = data['cleaned_text']

# Use CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=None)
X_count = count_vectorizer.fit_transform(documents)

# View Glossary
print("CountVectorizer Vocabulary:", count_vectorizer.get_feature_names_out())
print("CountVectorizer Feature Matrix:\n", X_count.toarray())

# Use TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# View Glossary
print("TfidfVectorizer Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TfidfVectorizer Feature Matrix:\n", X_tfidf.toarray())


"""
import numpy as np

# Converting a sparse matrix to a dense matrix
X_tfidf_dense = X_tfidf.toarray()

# Counting the number of non-zero elements
non_zero_count = np.count_nonzero(X_tfidf_dense)
print(f"Number of non-zero elements: {non_zero_count}")
"""


"""
Word Embeddings - Word2vec
"""


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


# Preprocessing - Tokenize the text
def preprocess_text(text):
    return word_tokenize(text.lower())

data['tokenized_text'] = data['cleaned_text'].apply(preprocess_text)

# View tokenized data
print(data['tokenized_text'].head())

# Train Word2Vec model
sentences = data['tokenized_text'].tolist()  # 将 DataFrame 转换为句子列表
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# View the vector for a specific word
word = 'dasani'
if word in model.wv:
    print(f"Vector for '{word}': {model.wv[word]}")
else:
    print(f"Word '{word}' not in vocabulary.")

# View similarity between two words
word1 = 'dasani'
word2 = 'water'
if word1 in model.wv and word2 in model.wv:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One of the words '{word1}' or '{word2}' not in vocabulary.")

# Find the most similar words to a specific word
most_similar = model.wv.most_similar('dasani', topn=5)
print(f"Words most similar to 'dasani': {most_similar}")

# Save the model
model.save("word2vec.model")

# # Load the model
# model = Word2Vec.load("word2vec.model")
