from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from process import pipeline
from train import train_baseline, train_model


if __name__ == '__main__':
    SEED = 10
    # FEATURES TO TEST
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

    # MODELS TO TEST
    models = {
        "MLP": MLPClassifier(max_iter=100, random_state=SEED),
        "BERNOULLI NB": BernoulliNB(),
        "LINEAR SVC": LinearSVC(random_state=SEED),
    }
    # CREATE DATA FOR TRAINING
    df1_train = pipeline("train.csv")
    df1_test = pipeline("test.csv")

    df2_train = pipeline("train.csv", ngrams=True)
    df2_test = pipeline("test.csv", ngrams=True)

    # BASELINE
    train_baseline(df1_train)

    # TRAIN & EVALUATE DIFFERENT MODELS ON DIFFERENT FEATURES
    for name, model in models.items():
        print("\n" + name + ":")
        print("\nOnly sentiment:")
        train_model(model, df1_train, df1_test, sentiment)

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