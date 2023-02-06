import spacy
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sentiment_analysis import return_polarity_score
from lexical_features import return_features, count_features

nlp = spacy.load("en_core_web_sm")
pd.set_option('display.max_columns', None)


def pipeline(file_name: str, ngrams=False):
    assert file_name in ["train.csv", "dev.csv", "test.csv"]

    raw_df = pd.read_csv(f"data/{file_name}")
    # Delete sentences with len <= 1
    raw_df = raw_df[raw_df["text"].apply(lambda x: len(x.split()) > 1)]

    if file_name == "train.csv":
        # Retrieve most meaningful keywords and ngrams
        global label2keywords, label2ngrams
        label2keywords = return_features(raw_df, 20)
        label2ngrams = return_features(raw_df, 20, ngrams=True)

    # Extract features
    features = [extract_features(sent) for sent in raw_df["text"]]

    # Create dataframe with features and labels
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

    # Extract tf idf
    fitted = None if file_name == "train.csv" else (vectorizer, transformer)
    tfidf_df = create_tfidf_df(raw_df, fitted, ngrams)

    return pd.concat([df, tfidf_df], axis=1)


def preprocess(sent):
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    return lemmata


def extract_features(sent):
    # Sentiment score
    sentiment = return_polarity_score(preprocess(sent))
    # Keyword counts
    kw_matches = [count_features(preprocess(sent), kw_list)
                  for kw_list in label2keywords.values()]
    # Ngram counts
    ngram_matches = [count_features(preprocess(sent), ngram_list)
                     for ngram_list in label2ngrams.values()]
    feat_list = [sentiment]
    feat_list.extend(kw_matches)
    feat_list.extend(ngram_matches)

    return feat_list


def create_tfidf_df(data, fitted=None, ngrams=False):
    # Create new vectorizer and transformer for training data
    if not fitted:
        global vectorizer, transformer
        ngram_range = (3, 3) if ngrams else (1, 1)
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        transformer = TfidfTransformer()

        counts = vectorizer.fit_transform(data["text"].tolist())
        tfidf = transformer.fit_transform(counts)

    # Use previously fitted vectorizer and transformer on test data
    else:
        vectorizer, transformer = fitted
        counts = vectorizer.transform(data["text"].tolist())
        tfidf = transformer.transform(counts)

    return pd.DataFrame.sparse.from_spmatrix(tfidf)