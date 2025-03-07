import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from xgboost import XGBClassifier


@pytest.fixture
def prepare_data():
    data_path1 = os.path.expanduser(
        "~/oumaima-benhaj-4DS6-ml_project/churn-bigml-20.csv"
    )
    data_path2 = os.path.expanduser(
        "~/oumaima-benhaj-4DS6-ml_project/churn-bigml-80.csv"
    )

    df_20 = pd.read_csv(data_path1)
    df_80 = pd.read_csv(data_path2)
    dfm = pd.concat([df_80, df_20], ignore_index=True)

    dfm["International plan"] = dfm["International plan"].map({"Yes": 1, "No": 0})
    dfm["Voice mail plan"] = dfm["Voice mail plan"].map({"Yes": 1, "No": 0})

    encoder = LabelEncoder()
    dfm["State"] = encoder.fit_transform(dfm["State"])

    X = dfm.drop("Churn", axis=1)
    y = dfm["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


@pytest.fixture
def train_model(prepare_data):
    x_train_scaled, _, y_train, _ = prepare_data

    reg = XGBClassifier(
        base_score=0.5,
        booster="gbtree",
        n_estimators=100,
        early_stopping_rounds=10,
        objective="binary:logistic",
        max_depth=3,
        learning_rate=0.3,
        eval_metric="logloss",
    )

    reg.fit(
        x_train_scaled, y_train, eval_set=[(x_train_scaled, y_train)], verbose=False
    )

    return reg


def test_evaluate_model(train_model, prepare_data):
    reg = train_model
    _, x_test_scaled, _, y_test = prepare_data

    y_pred = reg.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    assert acc > 0.5, "L'exactitude est trop basse"

    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)


def test_save_model(train_model):
    filename = "model.pkl"
    joblib.dump(train_model, filename)
    assert os.path.exists(filename), "Le fichier du modèle n'a pas été sauvegardé"
    print(f"Model saved as {filename}")


def test_load_model():
    filename = "model.pkl"
    if not os.path.exists(filename):
        pytest.skip(f"Le fichier {filename} n'existe pas. Test ignoré.")

    model = joblib.load(filename)
    assert model is not None, "Le modèle n'a pas été chargé correctement"
    print(f"Model loaded from {filename}")


if __name__ == "__main__":
    x_train_scaled, x_test_scaled, y_train, y_test = prepare_data()
    model = test_train_model(x_train_scaled, y_train)
    results = test_evaluate_model(model, x_test_scaled, y_test)

    if results:
        print(f"Accuracy: {results['accuracy']:.2f}")
        print("Classification Report:")
        print(results["report"])
    else:
        print("Erreur : Évaluation du modèle échouée.")
