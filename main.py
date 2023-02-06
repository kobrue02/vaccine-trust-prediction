from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from process import pipeline

def create_data(df, features, drop):

    test_set = pipeline("test.csv", "text")

    if features:
        # Drop specified features
        if drop:
            features.append("label")
            X_train = df.drop(features, axis=1)
            X_test = test_set.drop(features, axis=1)
        # Use specified features
        else:
            X_train = df[features]
            X_test = test_set[features]

    # Use all features
    else:
        X_train = df.drop("label", axis=1)
        X_test = test_set.drop("label", axis=1)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    y_train = df["label"]
    y_test = test_set["label"]

    # Create training and test sets
    
    return X_train, X_test, y_train, y_test

def train_baseline(df):
    X, y = create_data(df)
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)
    print(f"Acc of dummy baseline: {dummy_clf.score(X, y)}")

def train_model(model, df, features=None, drop=False):
    # Create training/test split and choose features to train model with
    X_train, X_test, y_train, y_test = create_data(df, features, drop)
    # Train model
    model.fit(X_train, y_train)
    # Make predictions
    y_predicted = model.predict(X_test)
    # Evaluate
    print(classification_report(y_test, y_predicted, zero_division=0))


if __name__ == '__main__':
    train_baseline(df1)
    SEED = 10
    sentiment = ["sentiment"]
    kw_matches = [
        "kw matches for label 0",
        "kw matches for label 1",
        "kw matches for label 2",
        "kw matches for label 3"
    ]
    ngram_matches = [
        "ngram matches for label 0",
        "ngram matches for label 1",
        "ngram matches for label 2",
        "ngram matches for label 3"
    ]
    own_features = sentiment + kw_matches + ngram_matches
    models = {
        "MLP": MLPClassifier(max_iter=100, random_state=SEED),
        "Bernoulli NB": BernoulliNB(),
        # "Gaussian NB": GaussianNB(),
        "Linear SVC": LinearSVC(random_state=SEED),
        # "SVC": SVC(random_state=SEED)
    }
    for name, model in models.items():
        print("\n" + name + ":")
        print("\nOnly sentiment:")
        train_model(model, df1, ["sentiment"])

        print("\nOnly kw matches:")
        train_model(model, df1, kw_matches)

        print("\nOnly ngram matches:")
        train_model(model, df1, ngram_matches)

        print("\nAll own features:")
        train_model(model, df1, own_features)

        #print("\nOnly tf idf on text:")
        #train_model(model, df1, own_features, drop=True)

        #print("\ntf idf on text + own features:")
        #train_model(model, df1)

        #print("\nOnly tf idf on ngrams:")
        #train_model(model, df2, own_features, drop=True)

        #print("\ntf idf on ngrams + own features:")
        #train_model(model, df2)
