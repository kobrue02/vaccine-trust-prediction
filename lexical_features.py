import pandas as pd
import spacy
from re import match
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nlp = spacy.load("en_core_web_sm")


def in_blacklist(w):
    return bool(match(r"(.)*(\d)+|covid(.)*|vaccin(\w)*", w.lower()))


def create_corpus(df, filter):
    """
    Combines all sents of one label to one document and
    creates a corpus out of all document strings"""
    corp = []
    labels = list(set(df["label"]))
    for label in labels:
        doc = ""
        for sent in df.loc[df["label"] == label]["text"]:
            tokens = nlp(sent)
            if filter:
                doc += " " + " ".join([t.lemma_ for t in tokens
                                       if not in_blacklist(t.lemma_)])
            else:
                doc += " " + " ".join([t.lemma_ for t in tokens])
        corp.append(doc)
    return corp


def get_sorted_indexes(df, matrix):
    """
    Create dict with labels as keys and
    lists of indexes of sorted tfidf values as values"""
    sorted_tfidf = {}
    labels = list(set(df["label"]))
    for label in labels:
        tfidf_array = matrix.toarray()[label]
        sorted_tfidf[label] = [
            i for i, val in sorted(list(enumerate(tfidf_array)),
                                   key=lambda x: x[1],
                                   reverse=True)]
    return sorted_tfidf


def get_top_n_features(feat_list, sorted_idxs, n):
    """
    Create dict with labels as keys and
    lists of features sorted by tfidf as values"""
    top_features_dict = dict()
    for label in sorted_idxs.keys():
        top_features_dict[label] = [feat_list[i] for i in
                                    sorted_idxs.get(label)[:n + 1]]
    return top_features_dict


def return_features(df, n, filter=True, ngrams=False):
    """
    Takes dataframe and proceeeds through all
    steps of keyword extraction """
    # Create corpus
    corpus = create_corpus(df, filter)
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
    sorted_indexes = get_sorted_indexes(df, tfidf_matrix)
    top_features = get_top_n_features(features, sorted_indexes, n)
    #sorted_indexes = get_sorted_indexes(df, count_matrix)
    #top_features = get_top_n_features(features, sorted_indexes, n)

    return top_features


def count_features(sent, top_features_dict):
    """
    Count occurence of keywords in sents for each label"""
    counts = []
    for features in top_features_dict.values():
        c = 0
        for ft in features:
            c += 1 if ft in sent else 0
        counts.append(c)
    return counts


if __name__ == '__main__':
    # Read CSV
    my_df = pd.read_csv("data/train.csv")
    # Extract keywords
    #keywords = return_features(my_df, 20)
    #print(keywords)
    # Extract ngrams
    ngrams = return_features(my_df, 20, ngrams=True)
    print(ngrams)