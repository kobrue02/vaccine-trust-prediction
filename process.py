import spacy
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sentiment_analysis import return_polarity_score
from lexical_features import return_features, count_features

nlp = spacy.load("en_core_web_sm")

# Read raw data
pd.set_option('display.max_columns', None)
raw_df = pd.read_csv("data/train.csv")
# Delete rows with sent length <= 1
raw_df = raw_df[raw_df["text"].apply(lambda x: len(x.split()) > 1)]
# Retrieve most meaningful keywords and ngrams
label2keywords = return_features(raw_df, 20)
label2ngrams = return_features(raw_df, 20, ngrams=True)


def pipeline(file_name: str, fitted=None, ngrams=False):
    raw_df = pd.read_csv(f"data/{file_name}")
    raw_df = raw_df[raw_df["text"].apply(lambda x: len(x.split()) > 1)]

    features = [extract_features(sent) for sent in raw_df["text"]]

    df = pd.DataFrame(features, columns=[
        "sentiment",
        "kw matches for label 0",
        "kw matches for label 1",
        "kw matches for label 2",
        "kw matches for label 3",
        "ngram matches for label 0",
        "ngram matches for label 1",
        "ngram matches for label 2",
        "ngram matches for label 3"
    ])

    df["label"] = raw_df["label"].tolist()

    if not fitted:
        tfidf_df, vectorizer, transformer = create_tfidf_df(raw_df, fitted, ngrams)
        return pd.concat([df, tfidf_df], axis=1), vectorizer, transformer
    else:
        tfidf_df = create_tfidf_df(raw_df, fitted, ngrams)
        return pd.concat([df, tfidf_df], axis=1)


def preprocess(sent):
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    return lemmata


def create_tfidf_df(data, fitted=None, ngrams=False):
    # Create new vectorizer and transformer for training data
    if not fitted:
        if ngrams:
            vectorizer = CountVectorizer(ngram_range=(3, 3),
                                         stop_words='english')
        else:
            vectorizer = CountVectorizer(stop_words='english')
        transformer = TfidfTransformer()

        counts = vectorizer.fit_transform(data["text"].tolist())
        tfidf = transformer.fit_transform(counts)

        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf)
        return tfidf_df, vectorizer, transformer
    # Use previously fitted vectorizer and transformer on test data
    else:
        vectorizer, transformer = fitted

        counts = vectorizer.transform(data["text"].tolist())
        tfidf = transformer.transform(counts)

        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf)
        return tfidf_df


def extract_features(sent):
    # Sentiment score
    sentiment = return_polarity_score(preprocess(sent))
    # Keyword counts
    kw_matches = [count_features(sent, kw_list)
                  for kw_list in label2keywords.values()]
    # Ngram counts
    ngram_matches = [count_features(sent, ngram_list)
                     for ngram_list in label2ngrams.values()]
    feat_list = [sentiment]
    feat_list.extend(kw_matches)
    feat_list.extend(ngram_matches)
    return feat_list
