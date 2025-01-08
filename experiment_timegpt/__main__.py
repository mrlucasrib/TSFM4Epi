import logging
import os

from utils_exp import Args, MLExperimentFacade, get_parser
from nixtla import NixtlaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)


def main():
    logger.info("Initializing experiment")
    parser = get_parser()
    args = parser.parse_args(namespace=Args)

    # defaults to 
    if os.environ.get("NIXTLA_API_KEY") is None:
        raise ValueError("NIXTLA_API_KEY must be set in the environment")
    nixtla_client = NixtlaClient()

    nixtla_client.cross_validation
    #TODO: WIP
    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )
    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=ttm_predictor,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()