from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import accuracy_score


MODEL_VERSION = "v1"
THIS_DIR = Path(__file__).parent
DATA_PATH = THIS_DIR / "data.csv"
MODEL_PATH = THIS_DIR / f"model_{MODEL_VERSION}.joblib"


def load_data():
    """Load the CSV into text (X) and labels (y)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    # Simple check
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("data.csv must have 'text' and 'label' columns")

    return df["text"], df["label"]


def build_pipeline():
    """
    Build a scikit-learn pipeline:
    - TF-IDF vectorizer: converts text to numeric features
    - LogisticRegression: classifier for sentiment
    """
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return pipeline


def train():
    # Load data
    X, y = load_data()


    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,  # keeps label distribution similar in both splits
    )

    # Build model
    model = build_pipeline()

    # Fit model
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluate on validation set
    print("\nValidation metrics:")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    val_accuracy = accuracy_score(y_val, y_pred)

    #Save model artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH.resolve()}")
    json_path = MODEL_PATH.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "model_version": MODEL_VERSION,
            "training_time": training_time,  # Replace with actual training time in seconds
            "datasize": len(X),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "validation_accuracy": val_accuracy
        }, f)



if __name__ == "__main__":
    train()
