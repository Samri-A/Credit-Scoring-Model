import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
sys.path.append("./src")
from train import train
from sklearn.datasets import make_classification

# Sample dummy DataFrame for mock
def generate_mock_df():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(5)])
    df["is_high_risk"] = y
    return df

@patch("train_script.mlflow")
@patch("train_script.os.makedirs")
@patch("train_script.mlflow.sklearn.save_model")
@patch("train_script.GridSearchCV.fit")
@patch("train_script.cross_val_score")
@patch("train_script.my_pipeline")
@patch("train_script.load_data")
def test_train_function(
    mock_load_data,
    mock_pipeline,
    mock_cross_val_score,
    mock_grid_fit,
    mock_save_model,
    mock_makedirs,
    mock_mlflow
):
    # Mock the data loading
    mock_df = generate_mock_df()
    mock_load_data.return_value = mock_df

    # Mock pipeline
    mock_pipeline.fit_transform.return_value = mock_df
    mock_pipeline.named_steps = {
        "columntransformer": MagicMock(get_feature_names_out=lambda: list(mock_df.columns))
    }

    # Mock cross_val_score to return consistent accuracy
    mock_cross_val_score.return_value = np.array([0.75] * 5)

    # Mock GridSearchCV fitting and best_estimator_
    mock_estimator = MagicMock()
    mock_estimator.predict.return_value = mock_df["is_high_risk"]
    mock_estimator.predict_proba.return_value = np.tile([0.2, 0.8], (100, 1))
    mock_grid = MagicMock(best_estimator_=mock_estimator, best_params_={"n_estimators": 50})
    mock_grid_fit.return_value = mock_grid

    # Run training
    train()

    # Assertions
    mock_load_data.assert_called_once()
    mock_pipeline.fit_transform.assert_called_once()
    mock_cross_val_score.assert_called()
    mock_save_model.assert_called()
    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_metric.called
    assert mock_mlflow.log_params.called

