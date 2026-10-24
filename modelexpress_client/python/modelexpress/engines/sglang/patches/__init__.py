# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime compatibility patches for SGLang."""

from ...vllm.patches.patch_hf_snapshot_prefetch import patch_hf_snapshot_prefetch
from .patch_object_storage_format_check import (
    patch_sglang_object_storage_format_check,
)
from .patch_sglang_metadata_prefetch import patch_sglang_metadata_prefetch

PATCHES = (
    patch_sglang_object_storage_format_check,
    patch_hf_snapshot_prefetch,
    patch_sglang_metadata_prefetch,
)

__all__ = ["PATCHES"]
