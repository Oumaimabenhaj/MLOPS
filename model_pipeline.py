import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from xgboost import XGBClassifier


def test_prepare_data():
    """Prépare les données pour l'entraînement."""
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
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "feature_names.joblib")
    y = dfm["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")

    return x_train_scaled, x_test_scaled, y_train, y_test


def test_train_model(x_train_scaled, y_train):
    """Entraîne le modèle XGBoost."""
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
        x_train_scaled,
        y_train,
        eval_set=[(x_train_scaled, y_train)],
        verbose=False,
    )

    return reg


def test_evaluate_model(reg, x_test_scaled, y_test):
    """Évalue le modèle et retourne les métriques de performance."""
    if reg is None or x_test_scaled is None or y_test is None:
        return None

    y_pred = reg.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)

    return {"accuracy": acc, "report": report}


def test_save_model(reg, filename="model.pkl"):
    """Sauvegarde le modèle dans un fichier pickle."""
    joblib.dump(reg, filename)
    print(f"Model saved as {filename}")


def test_load_model(filename="model.pkl"):
    """Charge un modèle sauvegardé."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")

    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


if __name__ == "__main__":
    x_train_scaled, x_test_scaled, y_train, y_test = test_prepare_data()
    model = test_load_model("xgboost_model.pkl")
    results = test_evaluate_model(model, x_test_scaled, y_test)

    if results:
        print(f"Accuracy: {results['accuracy']:.2f}")
        print("Classification Report:")
        print(results["report"])
    else:
        print("Erreur : Évaluation du modèle échouée.")
