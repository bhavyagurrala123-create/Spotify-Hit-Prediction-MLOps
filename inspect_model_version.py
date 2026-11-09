import mlflow
from mlflow import MlflowClient

client = MlflowClient()

model_name = "Spotify_Hit_Production_Model"
version = "1"

mv = client.get_model_version(model_name, version)

print("MODEL NAME:", mv.name)
print("VERSION:", mv.version)
print("STAGE:", mv.current_stage)
print("RUN ID:", mv.run_id)
print("SOURCE:", mv.source)
print("STATUS:", mv.status)
print("CREATION TIMESTAMP:", mv.creation_timestamp)