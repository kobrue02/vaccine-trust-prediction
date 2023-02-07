from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


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


def train_model(model, df_train, df_test, features=None, drop=False, display=True):
    # Create training & test data and choose features to train model with
    X_train, y_train = create_data(df_train, features, drop)
    X_test, y_test = create_data(df_test, features, drop)
    # Train model
    model.fit(X_train, y_train)
    # Make predictions
    y_predicted = model.predict(X_test)
    # Evaluate
    if display:
        print(classification_report(y_test, y_predicted, zero_division=0))