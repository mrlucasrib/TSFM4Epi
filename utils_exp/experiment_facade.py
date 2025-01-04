from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.client
import mlflow.experiments
import pandas as pd
from gluonts.dataset.arrow import ParquetFile
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import _to_dataframe

from utils_exp.splitter import TSFMExperimentSplitter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gluonts.dataset.common import Dataset
    from gluonts.dataset.split import AbstractBaseSplitter, TestData
    from gluonts.model.forecast import Forecast
    from gluonts.model.predictor import Predictor
    from pandas import DataFrame

import time

logger = logging.getLogger(__name__)


class MLExperimentFacade:
    def __init__(
        self,
        experiment_name: str,
        artifacts_path: str,
        prediction_length: int,
        context_length: int,
        splitter: AbstractBaseSplitter | None = None,
    ):
        self.experiment_name = experiment_name
        self.artifacts_path = artifacts_path
        self.context_length = context_length
        self.splitter = splitter or TSFMExperimentSplitter(
            context_length=context_length
        )
        self.prediction_length = prediction_length
        if (exp := mlflow.get_experiment_by_name(self.experiment_name)) is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifacts_path
            )
        else:
            self.experiment_id: str = exp.experiment_id

    def run_experiment(
        self, model_name: str, dataset_path, predictor: Predictor, num_samples: int
    ) -> None:
        mlflow.start_run(run_name=model_name, experiment_id=self.experiment_id)
        logger.info(f"Starting experiment {model_name}")
        mlflow.log_params(self.__dict__)
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_path": dataset_path,
                "num_samples": num_samples,
            }
        )

        data = self.get_data(dataset_path)
        logger.info(f"Data loaded from {dataset_path}")

        agg_metrics, item_metrics = self.backtest_metrics(
            data, predictor, num_samples=num_samples
        )
        logger.info(f"Metrics calculated")
        for metric_name, metric_value in agg_metrics.items():
            sanitized_metric_name = metric_name.replace("[", "_").replace("]", "_")
            mlflow.log_metric(sanitized_metric_name, metric_value)
        self.save_metrics(agg_metrics, f"{self.artifacts_path}/metrics.csv", model_name)
        logger.info(f"Metrics logged")

        item_metrics_path = f"{self.artifacts_path}/item_metrics.csv"
        item_metrics.to_csv(item_metrics_path, index=False)
        mlflow.log_artifact(item_metrics_path)
        mlflow.end_run()

    def get_data(self, path: str) -> ParquetFile:
        return ParquetFile(Path(path))

    def backtest_metrics(
        self,
        dataset: Dataset,
        predictor: Predictor,
        evaluator=Evaluator(quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        num_samples: int = 100,
    ) -> tuple[dict[str, float], DataFrame]:
        """
        Calculate metrics for a predictor.

        This method was implemented because the default method in the GluonTS library assumes splitters that do not meet the requirements of this evaluation.

        Parameters
        ----------
        dataset
            Dataset to use for testing.
        predictor
            The predictor to test.
        evaluator
            Evaluator to use.
        num_samples
            Number of samples to use when generating sample-based forecasts. Only
            sampling-based models will use this.

        Returns
        -------
        A tuple of aggregate metrics and metrics per time series split.
        """
        forecast_it, ts_it = self.make_evaluation_predictions(
            predictor=predictor, dataset=dataset, num_samples=num_samples
        )
        # Calculate time taken to make forecast
        start_time = time.time()
        forecast = list(forecast_it)
        end_time = time.time()
        logger.info(
            f"Time taken to transform to list of size {len(forecast)}: {end_time - start_time} seconds"
        )
        mlflow.log_metric("time_to_transform_forecast", end_time - start_time)
        mlflow.log_metric("forecast_size", len(forecast))

        agg_metrics, item_metrics = evaluator(ts_it, forecast)
        return agg_metrics, item_metrics

    def make_evaluation_predictions(
        self,
        predictor: Predictor,
        dataset: Dataset,
        num_samples: int,
        distance: int | None = None,
    ) -> tuple[Iterator[Forecast], Iterator[DataFrame]]:
        """
        Generate forecasts and time series from a dataset.

        This method was implemented because the default method in the GluonTS library assumes splitters that do not meet the requirements of this evaluation.

        Parameters
        ----------
        predictor
            Predictor to use.
        dataset
            Dataset to generate forecasts from.
        num_samples
            Number of samples to use when generating sample-based forecasts.
        distance
            Distance between windows. The default is the context length.

            Returns
            -------
            A tuple of forecasts and time series obtained by predicting `dataset` with `predictor`.
        """
        _, test_template = self.splitter.split(dataset)
        test_data: TestData = test_template.generate_instances(
            prediction_length=self.prediction_length,
            distance=distance,
        )
        return (
            predictor.predict(test_data.input, num_samples=num_samples),
            map(_to_dataframe, test_data),
        )

    def save_metrics(self, metrics: dict[str, float], path: str, model_name: str) -> None:
        """
        Save metrics to a CSV file using built-in Python libraries.

        Parameters
        ----------
        metrics : dict
            Dictionary containing metrics to save
        path : str
            Path where to save the CSV file
        model_name : str
            Name of the model used
        """
        
        # Check if file exists
        file_exists = os.path.exists(path)
        
        with open(path, 'a') as f:
            # Write header only if file doesn't exist
            if not file_exists:
                f.write('model_name,input_size,prediction_size,metric,value\n')
            for metric, value in metrics.items():
                sanitized_metric = metric.replace("[", "_").replace("]", "_")
                f.write(f'{model_name},{self.context_length},{self.prediction_length},{sanitized_metric},{value}\n')