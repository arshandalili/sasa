import argparse

import torch
import yaml
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, LoggingConfig
from sae_lens.registry import get_sae_training_class

import paths
import sasa  # noqa: F401  registers SASA with sae_lens
from configs import sae_config

SASA_FIELDS = ("gate", "k_aux", "aux_coefficient", "nuclear_coefficient", "nuclear_target", "encoder_norm_renorm")


def training_config(config: dict):
    config_class = get_sae_training_class(config["architecture"])[1]
    common = dict(d_in=config["d_in"], d_sae=config["columns"], normalize_activations=config["normalize_activations"],
                  apply_b_dec_to_input=config["apply_b_dec_to_input"])
    if "rank" in config:
        rank = config["rank"]
        return config_class(**common, n_groups=config["columns"] // rank, group_rank=rank,
                            k_groups=config["l0"] // rank, **{field: config[field] for field in SASA_FIELDS})
    common.update(aux_loss_coefficient=config["aux_loss_coefficient"],
                  rescale_acts_by_decoder_norm=config["rescale_acts_by_decoder_norm"])
    if config["architecture"] == "topk":
        return config_class(**common, k=int(config["l0"]))
    if config["architecture"] == "matryoshka_batchtopk":
        common.update(matryoshka_widths=[config["columns"] * quarter // 4 for quarter in (1, 2, 3, 4)])
    return config_class(**common, k=float(config["l0"]), topk_threshold_lr=config["topk_threshold_lr"])


def runner_config(config: dict, out, activation_cache) -> LanguageModelSAERunnerConfig:
    steps = config["training_tokens"] // config["train_batch_size_tokens"]
    return LanguageModelSAERunnerConfig(
        sae=training_config(config), model_name=config["model_name"], hook_name=config["hook_name"],
        dataset_path=config["dataset_path"], is_dataset_tokenized=config["is_dataset_tokenized"], streaming=True,
        context_size=config["context_size"], use_cached_activations=activation_cache is not None,
        cached_activations_path=None if activation_cache is None else str(activation_cache),
        store_batch_size_prompts=config["store_batch_size_prompts"], n_batches_in_buffer=config["n_batches_in_buffer"],
        training_tokens=config["training_tokens"], train_batch_size_tokens=config["train_batch_size_tokens"],
        lr=config["lr"], lr_warm_up_steps=config["lr_warm_up_steps"],
        lr_decay_steps=int(steps * config["lr_decay_fraction"]), dead_feature_window=config["dead_feature_window"],
        adam_beta1=config["adam_beta1"], adam_beta2=config["adam_beta2"], seed=config["seed"], dtype=config["dtype"],
        device="cuda" if torch.cuda.is_available() else "cpu", logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=config["n_checkpoints"], checkpoint_path=str(out / "checkpoints"), output_path=str(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sae", help="<config>/<sae>, e.g. gpt2_l7/sasa_r6_l0120_w24576")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="change settings of the SAE, e.g. training_tokens=204800 for a 50-step run")
    args = parser.parse_args()
    config = sae_config(args.sae)
    for assignment in args.set:
        field, value = assignment.split("=", 1)
        if field not in config:
            raise SystemExit(f"unknown field {field}; the fields are {sorted(config)}")
        config[field] = yaml.safe_load(value)
    out = paths.CHECKPOINTS / args.sae
    if (out / "sae_weights.safetensors").exists():
        raise SystemExit(f"{out} already holds a model")
    activation_cache = paths.ACTIVATIONS / config["activation_cache"] if config["activation_cache"] else None
    if activation_cache is not None and not activation_cache.exists():
        raise SystemExit(f"activation cache {activation_cache} not found (python cache.py {args.sae.split('/')[0]})")
    torch.manual_seed(config["seed"])  # sae_lens does not seed the initialization
    LanguageModelSAETrainingRunner(runner_config(config, out, activation_cache)).run()


if __name__ == "__main__":
    main()
