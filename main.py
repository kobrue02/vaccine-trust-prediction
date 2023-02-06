from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier

from process import pipeline


def create_data(df, features=None, drop=False):
    if features:  # Drop specified features
        if drop:
            features.append("label")
            X = df.drop(features, axis=1)
        else:  # Use specified features
            X = df[features]
    else:  # Use all features
        X = df.drop("label", axis=1)
    X.columns = X.columns.astype(str)
    y = df["label"]
    return X, y


def train_baseline(df):
    X, y = create_data(df)
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)
    print(f"Acc of dummy baseline: {dummy_clf.score(X, y)}")


def train_model(model, df_train, df_test, features=None, drop=False):
    # Create training & test data and choose features to train model with
    X_train, y_train = create_data(df_train, features, drop)
    X_test, y_test = create_data(df_test, features, drop)
    # Train model
    model.fit(X_train, y_train)
    # Make predictions
    y_predicted = model.predict(X_test)
    # Evaluate
    print(classification_report(y_test, y_predicted, zero_division=0))


if __name__ == '__main__':
    df1_train, vectorizer, transformer = pipeline("train.csv")
    df1_test = pipeline("test.csv", (vectorizer, transformer))

    df2_train, vectorizer, transformer = pipeline("train.csv", ngrams=True)
    df2_test = pipeline("test.csv", (vectorizer, transformer), ngrams=True)

    train_baseline(df1_train)

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
        train_model(model, df1_train, df1_test, ["sentiment"])

        print("\nOnly kw matches:")
        train_model(model, df1_train, df1_test, kw_matches)

        print("\nOnly ngram matches:")
        train_model(model, df1_train, df1_test, ngram_matches)

        print("\nAll own features:")
        train_model(model, df1_train, df1_test, own_features)

        print("\nOnly tf idf on text:")
        train_model(model, df1_train, df1_test, own_features, drop=True)

        print("\ntf idf on text + own features:")
        train_model(model, df1_train, df1_test)

        print("\nOnly tf idf on ngrams:")
        train_model(model, df2_train, df2_test, own_features, drop=True)

        print("\ntf idf on ngrams + own features:")
        train_model(model, df2_train, df2_test)
