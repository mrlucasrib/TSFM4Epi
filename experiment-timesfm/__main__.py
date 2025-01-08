from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import mlflow
import timesfm
from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import Dataset
    from pandas import DataFrame

from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Convert the dataset to a pandas DataFrame
class TimesFMExperiment:
    def __init__(self, dataset: Dataset ):
       ...


    def _prepare_data(self) -> DataFrame:
        train_path = "data/sinan/newdata/ANIMBR_municipality_M_train.parquet"
        test_path = "data/sinan/newdata/ANIMBR_municipality_M_test.parquet"
        disease_dataset = load_dataset("parquet", data_files={"train": train_path, "test": test_path})
        df_train = disease_dataset["train"].to_pandas()
        df_test = disease_dataset["test"].to_pandas()
        return df_train, df_test

    def forecast(self):
        with mlflow.start_run(run_name=self.__class__.__name__):
            params = {"backend": "gpu", "per_core_batch_size": 32, "horizon_len": 100}
            mlflow.log_params(params)
            df, df_test = self._prepare_data()
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(**params),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m")
            )
            forecast_df = self.model.forecast_on_df(          
                inputs=df,
                freq="M",  # monthly
                value_name="y",
                num_jobs=-1,
                # forecast_context_len=25,
                # window_size=25,
            )
            metrics = self.evaluate(forecast_df, df_test)
            print(metrics)
            for disease, disease_metrics in metrics.items():
                for metric_name, metric_value in disease_metrics.items():
                    mlflow.log_metric(f"{disease}_{metric_name}", metric_value)
            forecast_df.to_csv("forecast.csv", index=False)
            forecast_df.to_parquet("forecast.parquet", index=False)
            mlflow.log_artifact("forecast.csv")
            # live.log_artifact("forecast.csv", "dataset", "Forecasted data csv" )
            # live.log_artifact("forecast.parquet", "dataset", "Forecasted data parquet" )
            # live.end()
        return forecast_df


    def evaluate(self, df, df_test) -> dict:
        print(df)
        # Ensure the columns exist in both dataframes
        # Merge the dataframes on unique id and ds columns
        merged_df = df_test.merge(df[['unique_id', 'ds', 'timesfm']], on=['unique_id', 'ds'], how='inner')
        
        # Calculate error metrics for each disease
        metrics = {}
        for disease in merged_df['unique_id'].unique():
            disease_df = merged_df[merged_df['unique_id'] == disease]
            rmse = root_mean_squared_error(disease_df['y'], disease_df['timesfm'])
            mae = mean_absolute_error(disease_df['y'], disease_df['timesfm'])
            mape = (abs((disease_df['y'] - disease_df['timesfm']) / disease_df['y'])).mean() * 100
            smape = (abs(disease_df['y'] - disease_df['timesfm']) / ((abs(disease_df['y']) + abs(disease_df['timesfm'])) / 2)).mean() * 100

            metrics[disease] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "smape": smape,
            }

            metrics['all'] = {
                "rmse": root_mean_squared_error(merged_df['y'], merged_df['timesfm']),
                "mae": mean_absolute_error(merged_df['y'], merged_df['timesfm']),
                "mape": (abs((merged_df['y'] - merged_df['timesfm']) / merged_df['y'])).mean() * 100,
                "smape": (abs(merged_df['y'] - merged_df['timesfm']) / ((abs(merged_df['y']) + abs(merged_df['timesfm'])) / 2)).mean() * 100,
            }

        return metrics

times = TimesFMExperiment(None).forecast()
# print(times)