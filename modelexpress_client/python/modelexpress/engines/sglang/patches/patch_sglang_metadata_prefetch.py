# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch model metadata from ModelExpress Server before SGLang resolves it."""

from __future__ import annotations

import logging
from typing import Any

from .... import model_prefetch

logger = logging.getLogger(__name__)


def _safe_ensure_metadata(
    model_path: str | None,
    revision: str | None = None,
    *,
    cache_directory: str | None = None,
) -> None:
    if not isinstance(model_path, str) or not model_path:
        return
    try:
        model_prefetch.ensure_metadata(
            model_path,
            revision,
            cache_directory=cache_directory,
        )
    except Exception as exc:
        # Let the engine's own config/tokenizer loader report the failure it
        # expects instead of replacing it with an unfamiliar ModelExpress error.
        logger.warning(
            "ModelExpress metadata prefetch failed for %s: %s", model_path, exc
        )


def _prefetch_server_args_metadata(server_args: Any) -> None:
    try:
        from sglang.srt.arg_groups.overrides import resolving_view

        cfg = resolving_view(server_args)
    except Exception:
        cfg = server_args

    cache_dir = getattr(cfg, "download_dir", None)
    model_revision = getattr(cfg, "revision", None)
    tokenizer_revision = getattr(cfg, "tokenizer_revision", None) or model_revision
    draft_revision = (
        getattr(cfg, "speculative_draft_model_revision", None) or model_revision
    )

    seen: set[tuple[str, str | None, str | None]] = set()
    for path, rev in (
        (getattr(cfg, "model_path", None), model_revision),
        (
            getattr(cfg, "tokenizer_path", None) or getattr(cfg, "model_path", None),
            tokenizer_revision,
        ),
        (getattr(cfg, "speculative_draft_model_path", None), draft_revision),
    ):
        if not isinstance(path, str) or not path:
            continue
        key = (path, rev, cache_dir)
        if key in seen:
            continue
        seen.add(key)
        _safe_ensure_metadata(path, rev, cache_directory=cache_dir)


def _patch_resolution_hook() -> bool:
    try:
        from sglang.srt.arg_groups import resolution_hooks
    except ImportError:
        return False

    existing = resolution_hooks._HOOKS.get("handle_model_source_paths", ())
    if any(getattr(fn, "__modelexpress_patched__", False) for fn in existing):
        return False

    def _prefetch_hook(server_args: Any, previous: Any) -> None:
        _prefetch_server_args_metadata(server_args)
        previous(server_args)

    _prefetch_hook.__modelexpress_patched__ = True
    resolution_hooks.register_resolution_hook("handle_model_source_paths")(
        _prefetch_hook
    )
    return True


def _patch_model_config_init() -> bool:
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except ImportError:
        return False

    original = ModelConfig.__init__
    if getattr(original, "__modelexpress_patched__", False):
        return False

    def patched(
        self: Any,
        model_path: str,
        *args: Any,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _safe_ensure_metadata(model_path, revision)
        original(self, model_path, *args, revision=revision, **kwargs)

    patched.__modelexpress_patched__ = True
    patched.__wrapped__ = original
    ModelConfig.__init__ = patched
    return True


def patch_sglang_metadata_prefetch() -> bool:
    """Install ModelExpress metadata prefetch hooks into SGLang's startup path.

    Unlike vLLM, which calls ``huggingface_hub.snapshot_download`` upfront when
    parsing engine arguments, SGLang resolves ``config.json`` and tokenizer
    files directly via ``AutoConfig.from_pretrained`` and
    ``AutoTokenizer.from_pretrained``. This patch hooks both the server-argument
    resolution pipeline (``handle_model_source_paths``) and
    ``ModelConfig.__init__`` so that non-weight repository files are installed
    into the local Hugging Face cache before ``transformers`` looks them up.
    """
    if not model_prefetch.is_enabled():
        return False

    hook_applied = _patch_resolution_hook()
    config_applied = _patch_model_config_init()
    return hook_applied or config_applied
