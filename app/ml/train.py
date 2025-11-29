from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# File Paths
THIS_DIR = Path(__file__).parent
DATA_PATH = THIS_DIR / "data.csv"
MODEL_PATH = THIS_DIR / "model.joblib"


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
    # 1. Load data
    X, y = load_data()

    # 2. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,  # keeps label distribution similar in both splits
    )

    # 3. Build model
    model = build_pipeline()

    # 4. Fit model
    print("Training model...")
    model.fit(X_train, y_train)

    # 5. Evaluate on validation set
    print("\nValidation metrics:")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))

    # 6. Save model artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train()
