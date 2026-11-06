import argparse

from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig

import paths
from configs import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="e.g. gpt2_l7")
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["activation_cache"]:
        raise SystemExit(f"{args.config} trains on activations computed during training")
    CacheActivationsRunner(CacheActivationsRunnerConfig(
        dataset_path=config["dataset_path"], model_name=config["model_name"], hook_name=config["hook_name"],
        d_in=config["d_in"], context_size=config["context_size"], training_tokens=config["training_tokens"],
        model_batch_size=config["store_batch_size_prompts"], dtype=config["dtype"], seed=config["seed"],
        model_from_pretrained_kwargs={"center_writing_weights": False},
        new_cached_activations_path=str(paths.ACTIVATIONS / config["activation_cache"]), buffer_size_gb=1.0)).run()


if __name__ == "__main__":
    main()
