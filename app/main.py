from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import joblib
from pathlib import Path

from app.db.session import SessionLocal
from app.db import models
from app.db.base import init_db
from app.schemas.sentiment import (
    PredictRequest, PredictResponse, FeedbackRequest,
    PredictionItem, StatsResponse, ModelInfo
)


MODEL_VERSION = "v1"
MODEL_PATH = Path(__file__).parent / "ml" / f"model_{MODEL_VERSION}.joblib"

app = FastAPI(title="ML Sentiment Service")


# DATABASE DEPENDENCY

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup_event():
    init_db()
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)

    db = SessionLocal()
    try:
        model_row = db.query(models.Model).filter_by(version=MODEL_VERSION).first()
        if not model_row:
            m = models.Model(
                version=MODEL_VERSION,
                algorithm="logistic_regression",
            )
            db.add(m)
            db.commit()
    finally:
        db.close()


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    clf = app.state.model

    # predict sentiment
    labels = clf.classes_
    proba = clf.predict_proba([payload.text])[0]
    max_idx = proba.argmax()

    sentiment = labels[max_idx]
    confidence = float(proba[max_idx])

    # fetch model from DB
    model_row = db.query(models.Model).filter_by(version=MODEL_VERSION).first()
    if not model_row:
        raise HTTPException(status_code=500, detail="Model metadata missing")

    # save prediction in DB
    pred = models.Prediction(
        text=payload.text,
        predicted=sentiment,
        confidence=confidence,
        model_id=model_row.id
    )
    db.add(pred)
    db.commit()

    # return response to client
    return PredictResponse(
        sentiment=sentiment,
        confidence=confidence,
        model_version=model_row.version
    )


@app.post("/api/v1/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    pred = db.query(models.Prediction).get(payload.prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    fb = models.Feedback(
        prediction_id=pred.id,
        correct_label=payload.correct_label
    )
    db.add(fb)
    db.commit()

    return {"status": "ok"}


@app.get("/api/v1/predictions", response_model=list[PredictionItem])
def list_predictions(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    items = (
        db.query(models.Prediction)
        .order_by(models.Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items


@app.get("/api/v1/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    from sqlalchemy import func

    total = db.query(func.count(models.Prediction.id)).scalar() or 0

    def count(label):
        return (
            db.query(func.count(models.Prediction.id))
            .filter(models.Prediction.predicted == label)
            .scalar()
            or 0
        )

    return StatsResponse(
        total_predictions=total,
        positive=count("positive"),
        negative=count("negative"),
        neutral=count("neutral"),
    )


@app.get("/api/v1/model", response_model=ModelInfo)
def model_info(db: Session = Depends(get_db)):
    m = db.query(models.Model).filter_by(version=MODEL_VERSION).first()
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(
        version=m.version,
        algorithm=m.algorithm,
        created_at=m.created_at.isoformat()
    )
