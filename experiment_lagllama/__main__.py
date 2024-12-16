import torch
from lag_llama.gluon.estimator import LagLlamaEstimator

from utils_exp.cli import get_parser
from utils_exp.dummy_module import create_dummy_gluonts_torch_module
from utils_exp.experiment_facade import MLExperimentFacade

if __name__ == "__main__":

    create_dummy_gluonts_torch_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction_length = 24

    ckpt = torch.load(
        "content/lag-llama/lag-llama.ckpt", map_location=device, weights_only=False
    )  # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    estimator = LagLlamaEstimator(
        ckpt_path="content/lag-llama/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=32,  # Should not be changed; this is what the released Lag-Llama model was trained with
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    if __name__ == "__main__":
        parser = get_parser()
        args = parser.parse_args()

        create_dummy_gluonts_torch_module()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        prediction_length = args.prediction_length

        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
            ckpt_path=args.ckpt_path,
            prediction_length=prediction_length,
            context_length=args.context_length,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=1,
            num_parallel_samples=args.num_samples,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        exp = MLExperimentFacade(
            experiment_name=args.experiment_name,
            artifacts_path=args.artifacts_path,
            prediction_length=prediction_length,
            context_length=args.context_length,
        )
        exp.run_experiment(
            args.model_name,
            args.data_path,
            predictor,
            args.num_samples,
        )
