import os
from pathlib import Path

cache_root = Path(os.environ.get("SASA_CACHE_ROOT", Path.home() / ".cache" / "sasa"))
os.environ["HF_HOME"] = str(cache_root / "hf")
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hf_transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf_hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TMPDIR"] = str(cache_root / "tmp")

import datasets.config

# datasets 3.x's torch formatter imports torchvision.io.VideoReader, gone in 0.27
datasets.config.TORCHVISION_AVAILABLE = False
