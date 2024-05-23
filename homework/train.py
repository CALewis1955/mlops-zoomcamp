import os
import pickle
import click
import numpy as np
from mlflow import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("hw2_nyc_taxi_duration")

mlflow.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
     with mlflow.start_run() as run:
        
        mlflow.set_tag("developer", "cal")
       

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        #
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        
        # fetch the auto logged parameters and metrics for ended run
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        
        client = MlflowClient()
        run_id = run.info.run_id
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/models",
            name='random-forest-regressor'
)



if __name__ == '__main__':
    run_train()