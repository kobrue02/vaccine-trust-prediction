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

from sentiment_analysis import return_polarity_score
from lexical_features import return_features, count_features
from ngrams import *

nlp = spacy.load("en_core_web_sm")


pd.set_option('display.max_columns', None)
df = pd.read_csv("data/train.csv")
# Create dict with labels as keys and lists of top keywords as values
top_keywords = return_features(df)
top_ngrams = return_features(df, ngrams=True)


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


data = [process(tex) for tex in df["text"].tolist() if len(tex.split()) > 1]

dataset = pd.DataFrame(data, columns=["text", "lemmata", "label", "sentiment", "bigrams",
                                      "kw matches for label 0",
                                      "kw matches for label 1",
                                      "kw matches for label 2",
                                      "kw matches for label 3",
                                      "ngram matches for label 0",
                                      "ngram matches for label 1",
                                      "ngram matches for label 2",
                                      "ngram matches for label 3"])


def train_model(model):
    # Ablation Study
    # Hyperpamarameter tuning 
    # (Voting Classifier) 

    X, y = dataset["bigrams"], dataset["label"]
    print(X.shape, y.shape)

    vectorizer = CountVectorizer(max_features=1500, min_df=0.001, max_df=0.9)
    tfidfconverter = TfidfTransformer()

    X = vectorizer.fit_transform(X).toarray()
    X = tfidfconverter.fit_transform(X).toarray()
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y, random_state=1)

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    print(classification_report(y_test, predicted))



if __name__ == "__main__":

    print(dataset.head())
    for model in [MLPClassifier(max_iter=100), BernoulliNB(), GaussianNB()]
        train_model(model)

