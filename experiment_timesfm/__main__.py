# Predictor based on https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/timesfm.ipynb
from __future__ import annotations

import logging
from os import path
from typing import Iterator

from gluonts.model.predictor import RepresentablePredictor
import timesfm



import numpy as np
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast

from utils_exp import Args, MLExperimentFacade, get_parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

class TimesFmPredictor(RepresentablePredictor):
    def __init__(
        self,
        tfm,
        prediction_length: int,
        ds_freq: str,
        *args,
        **kwargs,
    ):
        self.tfm = tfm
        self.prediction_length = prediction_length
        if self.prediction_length > self.tfm.horizon_len:
            self.tfm.horizon_len = ((self.prediction_length + self.tfm.output_patch_len - 1) // self.tfm.output_patch_len) * self.tfm.output_patch_len
            logger.info('Jitting for new prediction length.')
        self.freq = timesfm.freq_map(ds_freq)

    def predict(self, dataset, batch_size: int = 1024, **kwargs) -> Iterator[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(dataset, batch_size=batch_size)):
            context = []
            for entry in batch:
                arr = np.array(entry["target"])
                context.append(arr)
            freqs = [self.freq] * len(context)
            _, full_preds = self.tfm.forecast(context, freqs, normalize=True)
            full_preds = full_preds[:, 0:self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts: list[Forecast] = []
        for item, ts in zip(forecast_outputs, dataset):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.tfm.quantiles)),
                    start_date=forecast_start_date,
                ))

        return iter(forecasts)

def main():
    logger.info("Initializing experiment")
    parser = get_parser()

    args = parser.parse_args(namespace=Args)
    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )
    if args.model_path == "timesfm-1.0":
        tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    num_layers=50,
                    horizon_len=128,
                    context_len=2048,
                    use_positional_embedding=False,
                    output_patch_len=128,
                ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path="/models/timesfm-1.0/checkpoints",
                # huggingface_repo_id="google/timesfm-1.0-200m"),
            ))
    else:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path="/models/timesfm-2.0/checkpoints",
                # huggingface_repo_id="google/timesfm-2.0-500m-jax"),
                ))
    predictor = TimesFmPredictor(
        tfm=tfm,
        prediction_length=args.prediction_length,
        ds_freq="M",
    )
    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=predictor,
        num_samples=args.num_samples,
    )

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("Running main")
    main()