import argparse

from configs import sae_config, sae_names

parser = argparse.ArgumentParser()
parser.add_argument("config", help="gpt2_l7 or mistral_l8")
parser.add_argument("--columns", type=int, nargs="*", help="only the SAEs with these widths")
args = parser.parse_args()
for name in sae_names(args.config):
    if args.columns is None or sae_config(name)["columns"] in args.columns:
        print(name)
