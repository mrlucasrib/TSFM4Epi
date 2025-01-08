# based on https://github.com/abdulfatir/gift-eval/blob/9ad8d03e3292ecd869e23c03d6f285ed9ffa5370/notebooks/naive.ipynb

import inspect
from dataclasses import dataclass, field
from collections.abc import Iterator
import logging

from gluonts.time_feature import get_seasonality
import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    AutoARIMA,
    AutoETS,
    AutoTheta,
    AutoRegressive,
    AutoCES,
)

from utils_exp import Args, MLExperimentFacade, get_parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

@dataclass
class ModelConfig:
    quantile_levels: list[float] | None = None
    forecast_keys: list[str] = field(init=False)
    statsforecast_keys: list[str] = field(init=False)
    intervals: list[int] | None = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """


    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        quantile_levels: list[float] | None = None,
        imputation_method: MissingValueImputation = LastValueImputation(),
        max_length: int | None = None,
        batch_size: int = 1,
        parallel: bool = False,
        **model_params,
    ) -> None:
        self.ModelType = model_params["stats_model"]
        if self.ModelType is None:
            raise ValueError("stats_model must be provided")
        del model_params["stats_model"]

        super().__init__(prediction_length=prediction_length)

        if "season_length" in inspect.signature(self.ModelType.__init__).parameters:
            model_params["season_length"] = season_length

        self.freq = freq
        self.model = StatsForecast(
            models=[self.ModelType(**model_params)],
            freq=freq,
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=-1 if parallel else 1,
        )
        self.fallback_model = StatsForecast(
            # Fallback model when main model returns NaNs
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=-1 if parallel else 1,
        )
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.imputation_method = imputation_method
        self.batch_size = batch_size
        self.max_length = max_length

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        batch = {}
        for idx, entry in enumerate(dataset):
            assert entry["target"].ndim == 1, "only for univariate time series"
            assert (
                len(entry["target"]) >= 1
            ), "all time series should have at least one data point"

            if self.max_length is not None:
                entry["start"] += len(entry["target"][: -self.max_length])
                entry["target"] = entry["target"][-self.max_length :]

            target = np.asarray(entry["target"], np.float32)
            if np.isnan(target).any():
                target = target.copy()
                target = self.imputation_method(target)

            unique_id = (
                f"{entry['item_id']}_{str(forecast_start(entry))}_{str(len(batch))}"
            )
            start = entry["start"]
            batch[entry['item_id']] = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        start=start.to_timestamp(),
                        periods=len(target),
                        freq=start.freq,
                    ).to_numpy(),
                    "y": target,
                }
            )

            if len(batch) == self.batch_size:
                self.logger.info(f"Processing batch {idx // self.batch_size + 1}.")
                results = self.sf_predict(pd.concat(batch.values()))
                yield from self.yield_forecast(batch.keys(), results)
                batch = {}

        if len(batch) > 0:
            self.logger.info(f"Processing final batch.")
            results = self.sf_predict(pd.concat(batch.values()))
            yield from self.yield_forecast(batch.keys(), results)

        self.logger.info("Prediction completed.")

    def sf_predict(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals
        results = self.model.forecast(
            df=Y_df,
            h=self.prediction_length,
            **kwargs,
        )
        # replace nan results with fallback
        row_nan = results.isnull().values.any(axis=-1)
        if row_nan.any():
            self.logger.warning(
                f"Model returned NaNs for {row_nan.sum()} time series. Using fallback model."
            )
            nan_ids = results[row_nan].index.values
            nan_df = Y_df[Y_df["unique_id"].isin(nan_ids)]
            fallback_results = self.fallback_model.forecast(
                df=nan_df,
                h=self.prediction_length,
                **kwargs,
            )
            results = pd.concat(
                [
                    results[~results.index.isin(nan_ids)],
                    fallback_results,
                ]
            )

        return results

    def yield_forecast(
        self, item_ids, results: pd.DataFrame
    ) -> Iterator[QuantileForecast]:
        for idx in item_ids:
            prediction = results[results["unique_id"] == idx]
            forecast_arrays = []
            model_name = self.ModelType.__name__
            for key in self.config.statsforecast_keys:
                if key == "mean":
                    forecast_arrays.append(prediction.loc[:, model_name].to_numpy())
                else:
                    forecast_arrays.append(
                        prediction.loc[:, f"{model_name}-{key}"].to_numpy()
                    )

            yield QuantileForecast(
                forecast_arrays=np.stack(forecast_arrays, axis=0),
                forecast_keys=self.config.forecast_keys,
                start_date=prediction.ds.iloc[0].to_period(freq=self.freq),
                item_id=idx,
            )


class NaivePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``Naive`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = Naive


def main():
    logger.info("Initializing experiment")
    parser = get_parser()

    args = parser.parse_args(namespace=Args)

    season_length = get_seasonality('M')
    matching_models = {
        "naive": Naive,
        "seasonal_naive": SeasonalNaive,
        "auto_arima": AutoARIMA,
        "auto_ets": AutoETS,
        "auto_theta": AutoTheta,
        "auto_regressive": AutoRegressive,
        "auto_ces": AutoCES,
    }
    predictor = StatsForecastPredictor(
            args.prediction_length,
            max_length=args.prediction_length,
            season_length=season_length,
            freq='M',
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            batch_size=args.batch_size,
            stats_model=matching_models[args.model_name],
    )
    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )

    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=predictor,
        num_samples=args.num_samples,
    )

if __name__ == "__main__":
    main()