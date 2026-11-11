# Modified by OPPO in 2026 from HumanOmniV2 to export the caption-specific
# Qwen Omni module.
from .vlm_module import VLMBaseModule
from .qwenomni_module4caption import QwenOmniModule

__all__ = ["VLMBaseModule", "QwenOmniModule"]
