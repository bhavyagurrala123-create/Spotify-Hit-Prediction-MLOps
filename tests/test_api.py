import pytest

from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_TRACK = {
    "track_id": "TEST001",
    "artists": "Test Artist",
    "album_name": "Test Album",
    "track_name": "Test Track",
    "track_genre": "pop",
    "explicit": "No",
    "duration_ms": 210000,
    "danceability": 0.72,
    "energy": 0.65,
    "key": 5,
    "loudness": -6.5,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.12,
    "instrumentalness": 0.0,
    "liveness": 0.11,
    "valence": 0.6,
    "tempo": 118.0,
    "time_signature": 4,
}


SECOND_TRACK = {
    "track_id": "TEST002",
    "artists": "Test Artist",
    "album_name": "Test Album",
    "track_name": "Test Track",
    "track_genre": "acoustic",
    "explicit": "Yes",
    "duration_ms": 185000,
    "danceability": 0.41,
    "energy": 0.3,
    "key": 2,
    "loudness": -11.2,
    "mode": 0,
    "speechiness": 0.04,
    "acousticness": 0.78,
    "instrumentalness": 0.02,
    "liveness": 0.09,
    "valence": 0.35,
    "tempo": 96.5,
    "time_signature": 3,
}


# ---------------------------------------------------------
# 9.1 - Health Endpoint
# ---------------------------------------------------------

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ---------------------------------------------------------
# 9.2 - Metadata Endpoint
# ---------------------------------------------------------

def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Spotify Hit Prediction API"
    assert data["version"] == "1.0.0"

    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True

    assert data["model_name"] is not None
    assert data["model_uri"] is not None


# ---------------------------------------------------------
# 9.3 - Valid Single Prediction
# ---------------------------------------------------------

def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json=VALID_TRACK,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "hit_probability" in data

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["hit_probability"] <= 1.0


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Explicit Flag
# ---------------------------------------------------------

def test_invalid_prediction_explicit(client):
    payload = VALID_TRACK.copy()

    payload["explicit"] = "Maybe"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "explicit"


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Duration
# ---------------------------------------------------------

def test_invalid_prediction_duration(client):
    payload = VALID_TRACK.copy()

    payload["duration_ms"] = -5

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "duration_ms"


# ---------------------------------------------------------
# 9.4 - Missing Required Field
# ---------------------------------------------------------

def test_missing_prediction_field(client):
    payload = VALID_TRACK.copy()

    del payload["danceability"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "danceability"


# ---------------------------------------------------------
# 9.5 - Valid Batch Prediction
# ---------------------------------------------------------

def test_valid_batch_prediction(client):
    payload = {
        "tracks": [
            VALID_TRACK,
            SECOND_TRACK,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2

    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "hit_probability" in prediction

        assert prediction["prediction"] in [0, 1]

        assert 0.0 <= prediction["hit_probability"] <= 1.0


# ---------------------------------------------------------
# 9.6 - Invalid Batch Prediction
# ---------------------------------------------------------

def test_invalid_batch_prediction(client):
    invalid_track = SECOND_TRACK.copy()

    invalid_track["explicit"] = "Maybe"

    payload = {
        "tracks": [
            VALID_TRACK,
            invalid_track,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "tracks.1.explicit"


# ---------------------------------------------------------
# 9.7 - Multiple Batch Validation Errors
# ---------------------------------------------------------

def test_multiple_batch_validation_errors(client):
    first_track = VALID_TRACK.copy()
    second_track = SECOND_TRACK.copy()

    first_track["explicit"] = "Maybe"
    second_track["duration_ms"] = -5

    payload = {
        "tracks": [
            first_track,
            second_track,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "tracks.0.explicit" in fields
    assert "tracks.1.duration_ms" in fields


# ---------------------------------------------------------
# 9.8 - Empty Batch
# ---------------------------------------------------------

def test_empty_batch(client):
    payload = {
        "tracks": []
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "tracks"


# ---------------------------------------------------------
# 9.9 - Invalid Datatype
# ---------------------------------------------------------

def test_invalid_datatype(client):
    payload = VALID_TRACK.copy()

    payload["duration_ms"] = "ten"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "duration_ms"


# ---------------------------------------------------------
# 9.10 - Extreme Valid Numeric Values
# ---------------------------------------------------------

def test_extreme_numeric_values(client):
    payload = VALID_TRACK.copy()

    payload["duration_ms"] = 100
    payload["danceability"] = 10000.0
    payload["TotalCharges"] = "1000000.0"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["hit_probability"] <= 1.0

# ---------------------------------------------------------
# Stage 10.1 - Null / Corrupted Input
# ---------------------------------------------------------

def test_null_value(client):
    payload = VALID_TRACK.copy()

    payload["explicit"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "explicit"

# =========================================================
# STAGE 10 - INVALID / EXTREME / CORRUPTED INPUT TESTING
# =========================================================


# ---------------------------------------------------------
# 10.2 - Malformed JSON
# ---------------------------------------------------------

def test_malformed_json(client):
    malformed_json = """
    {
        "track_id": "TEST001",
        "explicit": "Yes"
    """

    response = client.post(
        "/predict",
        content=malformed_json,
        headers={
            "Content-Type": "application/json"
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.3 - Completely Wrong JSON Structure
# ---------------------------------------------------------

def test_wrong_json_structure(client):
    payload = [
        "this",
        "should",
        "be",
        "an",
        "object"
    ]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.4 - Null Numeric Value
# ---------------------------------------------------------

def test_null_numeric_value(client):
    payload = VALID_TRACK.copy()

    payload["danceability"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "danceability"


# ---------------------------------------------------------
# 10.5 - Null Required String Value
# ---------------------------------------------------------

def test_null_total_charges(client):
    payload = VALID_TRACK.copy()

    payload["TotalCharges"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "TotalCharges"


# ---------------------------------------------------------
# 10.6 - Wrong Datatype for Multiple Fields
# ---------------------------------------------------------

def test_multiple_invalid_datatypes(client):
    payload = VALID_TRACK.copy()

    payload["duration_ms"] = "ten years"
    payload["danceability"] = "fifty"
    payload["SeniorCitizen"] = "old"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "duration_ms" in fields
    assert "danceability" in fields


# ---------------------------------------------------------
# 10.7 - Invalid Batch Structure
# ---------------------------------------------------------

def test_invalid_batch_structure(client):
    payload = {
        "tracks": VALID_TRACK
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.8 - Null Batch Tracks
# ---------------------------------------------------------

def test_null_batch_tracks(client):
    payload = {
        "tracks": None
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.9 - Batch Containing Null Track
# ---------------------------------------------------------

def test_batch_with_null_track(client):
    payload = {
        "tracks": [
            VALID_TRACK,
            None,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.10 - Extreme Negative Numeric Values
# ---------------------------------------------------------

def test_extreme_negative_values(client):
    payload = VALID_TRACK.copy()

    payload["duration_ms"] = -1000000
    payload["danceability"] = -999999.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "duration_ms" in fields
    assert "danceability" in fields


# ---------------------------------------------------------
# 10.11 - Very Large Valid Batch
# ---------------------------------------------------------

def test_large_batch(client):
    tracks = []

    for i in range(20):
        track = VALID_TRACK.copy()

        track["track_id"] = f"TEST{i:03d}"

        tracks.append(track)

    payload = {
        "tracks": tracks
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 20

    assert len(data["predictions"]) == 20

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]

        assert (
            0.0
            <= prediction["hit_probability"]
            <= 1.0
        )