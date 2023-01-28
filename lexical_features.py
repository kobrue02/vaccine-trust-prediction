import pandas as pd
import spacy
from re import match
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")


def in_blacklist(w):
    return bool(match(r"(.)*(\d)+|covid(.)*|vaccin(\w)*", w.lower()))


def create_corpus(data):
    """
    Combines all sents of one label to one document and
    creates a corpus out of all document strings"""
    corp = []
    labels = list(set(df["label"]))
    for label in labels:
        doc = ""
        for sent in data.loc[df["label"] == label]["text"]:
            tokens = nlp(sent)
            doc += " " + " ".join([t.lemma_ for t in tokens
                                   if not in_blacklist(t.lemma_)])
        corp.append(doc)
    return corp


def get_sorted_indexes(matrix):
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
    for label in sorted_indexes.keys():
        top_features_dict[label] = [feat_list[i] for i in
                                    sorted_idxs.get(label)[:n + 1]]
    return top_features_dict


def count_kws(sent, top_features_dict):
    """
    Count occurence of keywords in sents for each label"""
    counts = []
    for keywords in top_features_dict.values():
        c = 0
        for kw in keywords:
            c += 1 if kw in sent else 0
        counts.append(c)
    return counts


if __name__ == '__main__':
    # Read CSV
    df = pd.read_csv("data/train.csv")
    # Create corpus with 3 docs, each doc represents one label
    corpus = create_corpus(df)
    # Extract TF IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()
    # Get sorted indexes
    sorted_indexes = get_sorted_indexes(tfidf_matrix)
    # Get sorted features
    top_features = get_top_n_features(features, sorted_indexes, 50)
    print(top_features)
    # Count keywords
    kw_counts = [count_kws(sent, top_features) for sent in df["text"]]
    print(kw_counts)
    print(len(kw_counts))