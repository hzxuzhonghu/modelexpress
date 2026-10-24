# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang integration for ModelExpress."""

from ... import configure_sglang_logging
from ...patches import apply_patches
from .adapter import (
    SglangAdapter,
    build_sglang_load_context,
    build_sglang_source_identity,
)
from .loader import MxModelLoader
from .patches import PATCHES as SGLANG_PATCHES

_sglang_registered = False


def register_modelexpress_sglang() -> None:
    """Register ModelExpress's SGLang runtime hooks and patches."""
    global _sglang_registered
    if _sglang_registered:
        return

    configure_sglang_logging()
    apply_patches(SGLANG_PATCHES)

    _sglang_registered = True


__all__ = [
    "MxModelLoader",
    "SglangAdapter",
    "build_sglang_load_context",
    "build_sglang_source_identity",
    "register_modelexpress_sglang",
]
