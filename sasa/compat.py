"""Shims for upstream API drift."""


def patch_rope_theta() -> None:
    """Re-expose ``rope_theta`` on MistralConfig."""
    from transformers import MistralConfig

    if hasattr(MistralConfig, "rope_theta"):
        return

    def rope_theta(self):
        params = getattr(self, "rope_parameters", None) or {}
        if "rope_theta" not in params:
            raise AttributeError("rope_theta")
        return params["rope_theta"]

    MistralConfig.rope_theta = property(rope_theta)
