from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.']

vectorizer = CountVectorizer(ngram_range=(3, 3))
transformer = TfidfTransformer()
count_matrix = vectorizer.fit_transform(corpus)
tfidf_matrix = transformer.fit_transform(count_matrix)
print(tfidf_matrix.toarray())


