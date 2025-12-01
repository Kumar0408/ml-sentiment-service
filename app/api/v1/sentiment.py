from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    correct_label: str

class PredictionItem(BaseModel):
    id: int
    text: str
    predicted: str
    confidence: float

    class Config:
        orm_mode = True

class StatsResponse(BaseModel):
    total_predictions: int
    positive: int
    negative: int
    neutral: int

class ModelInfo(BaseModel):
    version: str
    algorithm: str
    created_at: str
