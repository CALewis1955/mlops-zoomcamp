if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, lr = data
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    TRACKING_SERVER_HOST = 'http://mlflow:5000'
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    mlflow.sklearn.autolog(registered_model_name="Linear Regression")
    mlflow.set_experiment("number 7")
    with mlflow.start_run():

        lr = data[1]
        dv = data[0]
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for NYC taxi data")

       # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="lr_model",
            registered_model_name="Linear-regression",
        )

        
    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
