from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

if TYPE_CHECKING:
    from datasets import Dataset
    from pandas import DataFrame

class ChronosExperiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        with mlflow.start_run(run_name=self.__class__.__name__):
            print("Running Chronos experiment with config:", self.config)
            train_df, test_df = self._prepare_data()
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_df,
                id_column="item_id",
                timestamp_column="timestamp"
            )
            train_data.head()
            predictor = TimeSeriesPredictor(prediction_length=100).fit(
                train_data, presets="bolt_small",
            )
            predictions = predictor.predict(train_data)
            df_pred = predictions.to_data_frame().reset_index()
            print(df_pred.head())
            # predictor.plot(
            #     data=train_data,
            #     predictions=predictions,
            #     item_ids=train_data.item_ids[:2],
            #     max_history_length=200,
            # )
            metrics = self.evaluate(df_pred, test_df)
            for disease, disease_metrics in metrics.items():
                for metric_name, metric_value in disease_metrics.items():
                    mlflow.log_metric(f"{disease}_{metric_name}", metric_value)

    
    def _prepare_data(self) -> DataFrame:
        train_path = "data/sinan/newdata/ANIMBR_municipality_M_train.parquet"
        test_path = "data/sinan/newdata/ANIMBR_municipality_M_test.parquet"
        disease_dataset = load_dataset("parquet", data_files={"train": train_path, "test": test_path})
        df_train = disease_dataset["train"].to_pandas()
        df_test = disease_dataset["test"].to_pandas()
        df_train.rename(columns={"y": "target", "unique_id": "item_id", "ds": "timestamp"}, inplace=True)
        df_test.rename(columns={"y": "target", "unique_id": "item_id", "ds": "timestamp"}, inplace=True)
        return df_train, df_test

    def evaluate(self, df, df_test) -> dict:
        merged_df = df_test.merge(df[['item_id', 'timestamp', 'mean']], on=['item_id', 'timestamp'], how='inner')
        
        # Calculate error metrics for each disease
        metrics = {}
        for disease in merged_df['item_id'].unique():
            disease_df = merged_df[merged_df['item_id'] == disease]
            rmse = root_mean_squared_error(disease_df['target'], disease_df['mean'])
            mae = mean_absolute_error(disease_df['target'], disease_df['mean'])
            mape = (abs((disease_df['target'] - disease_df['mean']) / disease_df['target'])).mean() * 100
            smape = (abs(disease_df['target'] - disease_df['mean']) / ((abs(disease_df['target']) + abs(disease_df['mean'])) / 2)).mean() * 100

            metrics[disease] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "smape": smape,
            }

            metrics['all'] = {
                "rmse": root_mean_squared_error(merged_df['target'], merged_df['mean']),
                "mae": mean_absolute_error(merged_df['target'], merged_df['mean']),
                "mape": (abs((merged_df['target'] - merged_df['mean']) / merged_df['target'])).mean() * 100,
                "smape": (abs(merged_df['target'] - merged_df['mean']) / ((abs(merged_df['target']) + abs(merged_df['mean'])) / 2)).mean() * 100,
            }

        return metrics
if __name__ == "__main__":
    ChronosExperiment(config={}).run()