import pandas as pd
import spacy
from re import match
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nlp = spacy.load("en_core_web_sm")


def in_blacklist(w):
    return bool(match(r"(.)*(\d)+|covid(.)*|vaccin(\w)*", w.lower()))


def create_corpus(data):
    """
    Combines all sents of one label to one document and
    creates a corpus out of all document strings"""
    corp = []
    index2label = dict()
    for i, label in enumerate(list(set(data["label"]))):
        doc = ""
        for sent in data.loc[data["label"] == label]["text"]:
            tokens = nlp(sent)
            doc += " " + " ".join([t.lemma_ for t in tokens
                                   if not in_blacklist(t.lemma_)])

        corp.append(doc)
        index2label[i] = label
    return corp, index2label


def get_sorted_indexes(matrix):
    """
    Create dict with labels as keys and
    lists of indexes of sorted tfidf values as values"""
    sorted_indexes = []
    for i in range(len(matrix.toarray())):
        tfidf_array = matrix.toarray()[i]
        sorted_indexes.append(
            [i for i, val in sorted(list(enumerate(tfidf_array)),
                                    key=lambda x: x[1], reverse=True)]
        )
    return sorted_indexes


def get_top_n_features(features, sorted_idxs, n):
    """
    Create dict with labels as keys and
    lists of features sorted by tfidf as values"""
    top_features = []
    for idx_list in sorted_idxs:
        top_features.append([features[i] for i in idx_list[:n + 1]])
    return top_features


def return_features(df, n, ngrams=False):
    """
    Takes dataframe and proceeeds through all
    steps of keyword extraction """
    # Create corpus
    corpus, index2label = create_corpus(df)
    # Use ngrams
    if ngrams:
        vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    # Use words
    else:
        vectorizer = CountVectorizer(stop_words='english')
    transformer = TfidfTransformer()
    count_matrix = vectorizer.fit_transform(corpus)
    tfidf_matrix = transformer.fit_transform(count_matrix)
    features = vectorizer.get_feature_names_out()
    # Get sorted features
    sorted_indexes = get_sorted_indexes(tfidf_matrix)
    top_features = get_top_n_features(features, sorted_indexes, n)
    label2features = {label: top_features[i] for i, label in index2label.items()}
    return label2features


def count_features(sent, feature_list):
    """
    Count occurence of keywords in sents for each label"""
    return sum([sent.count(ft) for ft in feature_list])


if __name__ == '__main__':
    # Read CSV
    my_df = pd.read_csv("data/train.csv")
    # Extract keywords
    # keywords = return_features(my_df, 20)
    # print(keywords)
    # Extract ngrams
    ngrams = return_features(my_df, 20, ngrams=True)
    print(ngrams)
