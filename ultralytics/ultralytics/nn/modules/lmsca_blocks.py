"""Public LMSCA-YOLO names mapped to the exact Hugging Face implementation.

The authoritative source is the authorized dataset repository
``TLMHoang/custom_yolo_code_full_ver``. Its naming differs from the paper/request:

- ``SAGRI`` is the scale-aligned gated injection block.
- ``IRDCB`` and ``LDown`` are defined directly in ``ultralytics.nn.modules.block``.
- ``DeformableHeadConv`` is the deformable head refinement block.
- ``KVCompressedAttention`` is the KVCA implementation and its ``dwconv`` mode
  uses the depthwise key-compression path.
"""

from ultralytics.nn.modules.block import (
    DeformableHeadConv,
    IRDCB,
    KVCompressedAttention,
    LDown,
    SAGRI,
)

DeformConvBlock = DeformableHeadConv

__all__ = [
    "SAGRI",
    "IRDCB",
    "LDown",
    "DeformableHeadConv",
    "DeformConvBlock",
    "KVCompressedAttention",
]
