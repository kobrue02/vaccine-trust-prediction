import random
import spacy
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC

from sentiment_analysis import return_polarity_score
from lexical_features import return_features, count_features
from ngrams import *

nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_columns', None)
df = pd.read_csv("data/train.csv")
# Create dict with labels as keys and lists of top keywords as values
top_keywords = return_features(df, n=20)
top_ngrams = return_features(df, n=20, ngrams=True)


def process(sent: str):
    # Make word frequency list
    # Sentiment Score
    # n-grams
    # troll extraction (eg 1 word)
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    label = int(df.loc[df['text'] == sent]["label"])
    sentiment = return_polarity_score(sent)
    bigrams = " ".join(n_gram(sent, 3))
    kw_counts = count_features(sent, top_keywords)
    ngram_counts = count_features(sent, top_ngrams)
    result = [sent.lower(), lemmata, label, sentiment, bigrams]
    result.extend(kw_counts)
    result.extend(ngram_counts)
    return result


def create_tfidf_matrix():
    vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    transformer = TfidfTransformer()
    count_matrix = vectorizer.fit_transform(dataset["text"])
    tfidf_matrix = transformer.fit_transform(count_matrix)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
    return tfidf_df


data = [process(tex) for tex in df["text"].tolist() if len(tex.split()) > 1]

dataset = pd.DataFrame(data, columns=["text", "lemmata", "label", "sentiment",
                                      "bigrams",
                                      "kw matches for label 0",
                                      "kw matches for label 1",
                                      "kw matches for label 2",
                                      "kw matches for label 3",
                                      "ngram matches for label 0",
                                      "ngram matches for label 1",
                                      "ngram matches for label 2",
                                      "ngram matches for label 3"])

tfidf_matrix = create_tfidf_matrix()
new_dataset = pd.concat([dataset, tfidf_matrix], axis=1)

# Ablation study
sentiment_only = ["sentiment"]
kw_matches_only = ["kw matches for label 0",
                   "kw matches for label 1",
                   "kw matches for label 2",
                   "kw matches for label 3"]
ngram_matches_only = ["ngram matches for label 0",
                      "ngram matches for label 1",
                      "ngram matches for label 2",
                      "ngram matches for label 3"]


def train_baseline():
    pass


def create_data(features: list, drop):
    if drop:
        X = new_dataset.drop(features, axis=1)
    else:
        X = dataset[features]
    X.columns = X.columns.astype(str)
    y = new_dataset["label"]
    # Create training and test sets
    return train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=10)


def train_linear_svc(features: list, drop=False):
    X_train, X_test, y_train, y_test = create_data(features, drop)
    # Initialize model
    clf = LinearSVC(random_state=10)
    # Train model
    clf.fit(X_train, y_train)
    # Compute accuracy
    predicted = clf.predict(X_test)
    # print(f"Accuracy LinearSVC: {clf.score(X_test, y_test)}")
    print("Linear SVC:")
    print(classification_report(y_test, predicted, zero_division=0))


def train_svc(features: list, drop=False):
    X_train, X_test, y_train, y_test = create_data(features, drop)
    # Initialize model
    clf = SVC(random_state=10)
    # Train model
    clf.fit(X_train, y_train)
    # Compute accuracy
    predicted = clf.predict(X_test)
    # print(f"Accuracy SVC: {clf.score(X_test, y_test)}")
    print("SVC:")
    print(classification_report(y_test, predicted, zero_division=0))


def train_model():
    # Ablation Study
    # Hyperpamarameter tuning 
    # (Voting Classifier) 
    model = MLPClassifier(max_iter=100)

    X, y = dataset["text"], dataset["label"]
    # print(X.shape, y.shape)

    vectorizer = CountVectorizer(max_features=1500, min_df=0.001, max_df=0.9)
    tfidfconverter = TfidfTransformer()

    X = vectorizer.fit_transform(X).toarray()
    X = tfidfconverter.fit_transform(X).toarray()
    # print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,
                                                        stratify=y,
                                                        random_state=1)

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    print(classification_report(y_test, predicted))


if __name__ == "__main__":
    # print(dataset.head())
    # train_model()
    train_linear_svc(["text", "lemmata", "label", "bigrams"], drop=True)
    train_linear_svc(["text", "lemmata", "label", "bigrams",
                      "sentiment",
                      "kw matches for label 0",
                      "kw matches for label 1",
                      "kw matches for label 2",
                      "kw matches for label 3",
                      "ngram matches for label 0",
                      "ngram matches for label 1",
                      "ngram matches for label 2",
                      "ngram matches for label 3"], drop=True)
    train_svc(["text", "lemmata", "label", "bigrams"], drop=True)
    train_svc(["text", "lemmata", "label", "bigrams",
               "sentiment",
               "kw matches for label 0",
               "kw matches for label 1",
               "kw matches for label 2",
               "kw matches for label 3",
               "ngram matches for label 0",
               "ngram matches for label 1",
               "ngram matches for label 2",
               "ngram matches for label 3"], drop=True)

    # for i in range(0,4):
    #    raw_text = dataset[dataset["label"] == i]["text"].tolist()
    #    bigrams = most_common_ngrams(raw_text, 4)
    #    print(bigrams)
