from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrackInput(BaseModel):
    """One Spotify track described by its audio features (no popularity - that is what we predict)."""

    track_id: str

    # descriptive text - accepted for convenience, ignored by the model
    artists: Optional[str] = None
    album_name: Optional[str] = None
    track_name: Optional[str] = None

    track_genre: str = Field(min_length=1)
    explicit: Literal["Yes", "No"]

    duration_ms: int = Field(gt=0)
    danceability: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    key: int = Field(ge=0, le=11)
    loudness: float = Field(ge=-60.0, le=5.0)
    mode: Literal[0, 1]
    speechiness: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    instrumentalness: float = Field(ge=0.0, le=1.0)
    liveness: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    tempo: float = Field(ge=0.0, le=300.0)
    time_signature: int = Field(ge=0, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    hit_probability: float


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    tracks: list[TrackInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
