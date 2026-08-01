"""Shims for upstream API drift."""

import inspect


def patch_rope_theta() -> None:
    """Re-expose ``rope_theta`` on MistralConfig once it moves into ``rope_parameters``."""
    from transformers import MistralConfig

    if "rope_parameters" not in inspect.signature(MistralConfig.__init__).parameters:
        return

    def getter(self):
        params = getattr(self, "rope_parameters", None) or {}
        if "rope_theta" not in params:
            raise AttributeError("rope_theta")
        return params["rope_theta"]

    def setter(self, value):
        self.__dict__.setdefault("rope_parameters", {})["rope_theta"] = value

    MistralConfig.rope_theta = property(getter, setter)
