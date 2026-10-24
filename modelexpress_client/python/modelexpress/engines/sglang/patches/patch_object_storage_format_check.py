# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve SGLang's ModelExpress load format when model paths use object storage."""

from __future__ import annotations

from typing import Any


def _is_modelexpress_remote_instance(cfg: Any) -> bool:
    load_format = getattr(cfg, "load_format", None)
    load_format = getattr(load_format, "value", load_format)
    backend = getattr(cfg, "remote_instance_weight_loader_backend", None)
    return str(load_format) == "remote_instance" and str(backend) == "modelexpress"


def _resolving_cfg(server_args: Any) -> Any:
    try:
        from sglang.srt.arg_groups.overrides import resolving_view

        return resolving_view(server_args)
    except Exception:
        return server_args


def _declare_load_format_overrides(server_args: Any, **fields: Any) -> None:
    try:
        from sglang.srt.arg_groups.overrides import declare_resolution

        declare_resolution(
            server_args,
            "_modelexpress_object_storage_load_format",
            **fields,
        )
        return
    except Exception:
        pass

    for name, value in fields.items():
        setattr(server_args, name, value)


def patch_sglang_object_storage_format_check() -> bool:
    """Keep ``load_format='remote_instance'`` when ``model_path`` is an object-store URI.

    SGLang's ``handle_load_format`` unconditionally rewrites ``load_format``
    (and ``speculative_draft_load_format``) to ``"runai_streamer"`` whenever
    ``is_runai_obj_uri(model_path)`` is true, bypassing ``MxModelLoader`` even
    when the user explicitly configured ``--load-format remote_instance
    --remote-instance-weight-loader-backend modelexpress``. Wrap the
    ``handle_load_format`` resolution step so ModelExpress retains control of
    the load pipeline and streams from the object-storage URI on a P2P miss.
    """
    try:
        from sglang.srt.arg_groups import resolution_hooks
    except ImportError:
        return False

    existing = resolution_hooks._HOOKS.get("handle_load_format", ())
    if any(
        getattr(fn, "__modelexpress_object_storage_patched__", False)
        for fn in existing
    ):
        return False

    def _preserve_modelexpress_load_format(
        server_args: Any, previous: Any
    ) -> None:
        before_cfg = _resolving_cfg(server_args)
        preserve_main = _is_modelexpress_remote_instance(before_cfg)
        initial_draft_format = getattr(
            before_cfg, "speculative_draft_load_format", None
        )

        previous(server_args)

        if not preserve_main:
            return

        after_cfg = _resolving_cfg(server_args)
        updates: dict[str, Any] = {}
        current_load_format = getattr(after_cfg, "load_format", None)
        current_load_format = getattr(
            current_load_format, "value", current_load_format
        )
        if str(current_load_format) != "remote_instance":
            updates["load_format"] = "remote_instance"

        current_draft_format = getattr(
            after_cfg, "speculative_draft_load_format", None
        )
        current_draft_format = getattr(
            current_draft_format, "value", current_draft_format
        )
        if (
            initial_draft_format in (None, "remote_instance")
            and str(current_draft_format) == "runai_streamer"
        ):
            updates["speculative_draft_load_format"] = initial_draft_format

        if updates:
            _declare_load_format_overrides(server_args, **updates)

    _preserve_modelexpress_load_format.__modelexpress_object_storage_patched__ = True
    resolution_hooks.register_resolution_hook("handle_load_format")(
        _preserve_modelexpress_load_format
    )
    return True
