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
from lexical_features import return_keywords, count_kws

nlp = spacy.load("en_core_web_sm")


pd.set_option('display.max_columns', None)
df = pd.read_csv("data/train.csv")
# Create dict with labels as keys and lists of top keywords as values
keywords = return_keywords(df)


def process(sent):
    # Make word frequency list
    # Sentiment Score
    # n-grams
    # troll extraction (eg 1 word)
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    label = int(df.loc[df['text'] == sent]["label"])
    sentiment = return_polarity_score(sent)
    kw_counts = count_kws(sent, keywords)
    result = [sent, lemmata, label, sentiment]
    result.extend(kw_counts)
    return result


data = [process(tex) for tex in df["text"].tolist() if len(tex.split()) > 1]

dataset = pd.DataFrame(data, columns=["text", "lemmata", "label", "sentiment",
                                      "kw matches for label 0",
                                      "kw matches for label 1",
                                      "kw matches for label 2",
                                      "kw matches for label 3"])

# Ablation Study
# Hyperpamarameter tuning 
# (Voting Classifier)
model = MLPClassifier(max_iter=100)

X, y = dataset["lemmata"], dataset["label"]

vectorizer = CountVectorizer(max_features=1500, min_df=0.001, max_df=0.9)
tfidfconverter = TfidfTransformer()

X = vectorizer.fit_transform(X).toarray()
X = tfidfconverter.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y, random_state=1)

model.fit(X_train, y_train)

predicted = model.predict(X_test)
print(classification_report(y_test, predicted))

