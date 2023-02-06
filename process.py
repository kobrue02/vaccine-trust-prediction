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

def pipeline(file_name: str, type: str):

    assert type in ["text", "ngrams"]

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

    if type == "text":
        # Create dataframe based on own features and tf idf on text
        df1 = pd.concat([df, create_tfidf_df(raw_df)], axis=1)
        return df1
    
    if type == "ngrams":
        # Create dataframe based on own features and tf idf on ngrams
        df2 = pd.concat([df, create_tfidf_df(raw_df, ngrams=True)], axis=1)
        return df2


def preprocess(sent):
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    return lemmata


def create_tfidf_df(data, ngrams=False):
    if ngrams:
        vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')
    transformer = TfidfTransformer()
    count_matrix = vectorizer.fit_transform(data["text"].tolist())
    tfidf_matrix = transformer.fit_transform(count_matrix)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
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
