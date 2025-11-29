from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parent / "model.joblib"

def main():
    model = joblib.load(MODEL_PATH)

    texts = [
        "I absolutely love this!",
        "This is terrible, I hate it",
        "It's okay, nothing special"
    ]

    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    for text, label, proba in zip(texts, preds, probs):
        print("TEXT:", text)
        print("PRED:", label)
        print("PROBS:", proba)
        print()

if __name__ == "__main__":
    main()
