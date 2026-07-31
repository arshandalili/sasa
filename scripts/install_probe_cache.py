"""Copy a complete probe cache from an existing SAEBench install.

The probe and the train/test split are already in artifacts/absorption/probes/; use this
only to import a cache from elsewhere. Otherwise run scripts.rebuild_probe_activations.

    python -m scripts.install_probe_cache --source <sae_bench>/artifacts/absorption/probes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEST = REPO_ROOT / "artifacts" / "absorption" / "probes"

# The *_act_tensor.dat memmaps alongside these are never read back.
NEEDED = ("probe.pth", "data.npz", "train_df.csv", "test_df.csv")


def probe_subdir(model: str, hook: str, layer: int) -> Path:
    return Path(model) / hook.replace("/", "_").replace(".", "_") / f"layer_{layer}"


def install(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        src = src_dir / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        if name == "data.npz":
            data = np.load(src)
            np.savez_compressed(dest_dir / name, **{k: data[k] for k in data.files})
        else:
            shutil.copy2(src, dest_dir / name)
        print(f"  {name:16s} {(dest_dir / name).stat().st_size / 1e6:8.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="SAEBench probes directory, i.e. <sae_bench>/artifacts/absorption/probes")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--hook", default="blocks.7.hook_resid_pre")
    ap.add_argument("--layer", type=int, default=7)
    args = ap.parse_args()

    rel = probe_subdir(args.model, args.hook, args.layer)
    src = Path(args.source) / rel
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    print(f"{src} -> {DEST / rel}")
    install(src, DEST / rel)


if __name__ == "__main__":
    main()
