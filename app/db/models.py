from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .session import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    algorithm = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    predicted = Column(String)
    confidence = Column(Float)
    model_id = Column(Integer, ForeignKey("models.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("Model", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    correct_label = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction = relationship("Prediction", back_populates="feedback")
