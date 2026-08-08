import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack


def main():
    os.makedirs("model", exist_ok=True)

    df = pd.read_csv(os.path.join("data", "cleaned_data.csv"))

    df = df.dropna(subset=["cleaned_composition", "manufacturer", "price"])
    df = df[df["price"] > 0]
    df = df.reset_index(drop=True)

    le = LabelEncoder()
    df["manufacturer_enc"] = le.fit_transform(df["manufacturer"].astype(str))

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    comp_features = tfidf.fit_transform(df["cleaned_composition"])

    manuf_features = df[["manufacturer_enc"]].values
    X = hstack([comp_features, manuf_features])
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: Rs.{mae:.2f}")

    joblib.dump(model, os.path.join("model", "model.joblib"))
    joblib.dump(le, os.path.join("model", "encoder.joblib"))
    joblib.dump(tfidf, os.path.join("model", "tfidf.joblib"))
    df.to_csv(os.path.join("model", "model_data.csv"), index=False)

    print("Saved model.joblib, encoder.joblib, tfidf.joblib, model_data.csv to /model")


main()
