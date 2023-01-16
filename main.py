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

nlp = spacy.load("en_core_web_sm")


pd.set_option('display.max_columns', None)
df = pd.read_csv("train.csv")


def process(sent):
    # Make word frequency list
    # Sentiment Score
    # n-grams
    # troll extraction (eg 1 word)
    doc = nlp(sent)
    lemmata = str(" ".join([t.lemma_ for t in doc]))
    tags = str(" ".join([t.tag_ for t in doc]))
    label = int(df.loc[df['text'] == sent]["label"])
    return [sent, lemmata, tags, label]


data = [process(tex) for tex in df["text"].tolist()]

dataset = pd.DataFrame(data, columns=["text", "lemmata", "tags", "label"])

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

