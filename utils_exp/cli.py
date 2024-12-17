import argparse

class Args(argparse.Namespace):
    experiment_name: str
    artifacts_path: str
    prediction_length: int
    context_length: int
    ckpt_path: str
    model_name: str
    data_path: str
    num_samples: int
    model_size: str

    
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="artifacts",
        help="Path to save artifacts",
    )
    parser.add_argument(
        "--prediction_length", type=int, default=24, help="Prediction length"
    )
    parser.add_argument("--context_length", type=int, default=32, help="Context length")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint path",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples"
    )
    parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    help="Model size",
    )
    return parser


__all__ = ["get_parser"]
