from sae_lens import register_sae_class, register_sae_training_class

from sasa.model import SASA, SASAConfig, SASAInference, SASAInferenceConfig, TopKSASAConfig, TopKSASAInferenceConfig

# the architecture names stored in the released checkpoints
register_sae_training_class("batchtopk_manifold_sae", SASA, SASAConfig)
register_sae_class("batchtopk_manifold_sae", SASAInference, SASAInferenceConfig)
register_sae_training_class("topk_sasa", SASA, TopKSASAConfig)
register_sae_class("topk_sasa", SASAInference, TopKSASAInferenceConfig)
