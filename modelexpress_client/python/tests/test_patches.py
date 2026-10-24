# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from types import ModuleType, SimpleNamespace

from modelexpress.engines.vllm.patches import _version as vllm_patch_version
from modelexpress.engines.vllm.patches.patch_humming_regex_ignore import (
    patch_humming_regex_ignore,
)
from modelexpress.engines.vllm.patches.patch_object_storage_format_check import (
    patch_object_storage_format_check,
)
from modelexpress.patches import apply_patches


def test_apply_patches_runs_in_order(monkeypatch):
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    calls = []

    def first():
        calls.append("first")
        return False

    def second():
        calls.append("second")
        return True

    apply_patches((first, second))

    assert calls == ["first", "second"]


def test_apply_patches_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MX_DISABLE_PATCHES", "true")
    calls = []

    def patch():
        calls.append("patch")
        return True

    apply_patches((patch,))

    assert calls == []


def test_vllm_entrypoint_configures_logging_before_patches(monkeypatch):
    from modelexpress.engines import vllm as vllm_integration
    from modelexpress.engines.vllm import registration

    calls = []
    monkeypatch.setattr(vllm_integration, "_loaders_registered", False)
    monkeypatch.setattr(
        vllm_integration,
        "configure_vllm_logging",
        lambda: calls.append(("logging", None)),
    )
    monkeypatch.setattr(
        vllm_integration,
        "apply_patches",
        lambda patches: calls.append(("patches", patches)),
    )
    monkeypatch.setattr(
        registration,
        "register_plugin_model_loader",
        lambda: calls.append(("loader", None)),
    )
    monkeypatch.setattr(
        registration,
        "register_plugin_weight_transfer_engine",
        lambda: calls.append(("weight_transfer", None)),
    )

    vllm_integration.register_modelexpress_loaders()

    assert calls == [
        ("logging", None),
        ("patches", vllm_integration.VLLM_PATCHES),
        ("loader", None),
        ("weight_transfer", None),
    ]


def test_object_storage_patch_preserves_model_weights(monkeypatch):
    calls = []

    class VllmConfig:
        def try_verify_and_update_config(self):
            calls.append(hasattr(self.model_config, "model_weights"))

    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = VllmConfig
    transformers_module = ModuleType("vllm.transformers_utils")
    runai_module = ModuleType("vllm.transformers_utils.runai_utils")
    runai_module.is_runai_obj_uri = lambda uri: uri.startswith("s3://")
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(
        sys.modules,
        transformers_module.__name__,
        transformers_module,
    )
    monkeypatch.setitem(sys.modules, runai_module.__name__, runai_module)
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.21.0",
    )

    assert patch_object_storage_format_check() is True

    config = VllmConfig()
    config.load_config = SimpleNamespace(load_format="modelexpress")
    config.model_config = SimpleNamespace(model_weights="s3://bucket/model")
    config.try_verify_and_update_config()

    assert calls == [False]
    assert config.model_config.model_weights == "s3://bucket/model"
    assert patch_object_storage_format_check() is False


def test_object_storage_patch_skips_unknown_version(monkeypatch):
    class VllmConfig:
        def try_verify_and_update_config(self):
            pass

    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = VllmConfig
    transformers_module = ModuleType("vllm.transformers_utils")
    runai_module = ModuleType("vllm.transformers_utils.runai_utils")
    runai_module.is_runai_obj_uri = lambda uri: True
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(
        sys.modules,
        transformers_module.__name__,
        transformers_module,
    )
    monkeypatch.setitem(sys.modules, runai_module.__name__, runai_module)
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.22.0",
    )

    original = VllmConfig.try_verify_and_update_config
    assert patch_object_storage_format_check() is False
    assert VllmConfig.try_verify_and_update_config is original


class _BrokenHummingConfig:
    def get_from_keys_or(self, config, keys, default):
        return next((config[key] for key in keys if key in config), default)

    def is_layer_skipped(self, config, prefix):
        ignored = self.get_from_keys_or(
            config,
            ["ignored_layers", "ignore", "modules_to_not_convert"],
            [],
        )
        return any(module_name in prefix for module_name in ignored)


def _install_fake_humming(monkeypatch, version, config_cls):
    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: version,
    )

    humming = ModuleType("vllm.model_executor.layers.quantization.humming")
    humming.HummingConfig = config_cls
    humming.HummingMethod = object
    humming.re = re
    monkeypatch.setitem(sys.modules, humming.__name__, humming)


def test_humming_patch_adds_regex_and_preserves_substring_matching(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        pass

    _install_fake_humming(monkeypatch, "0.25.1", HummingConfig)

    assert patch_humming_regex_ignore() is True
    config = HummingConfig()
    assert config.is_layer_skipped(
        {"ignore": ["re:^vision_tower.*"]},
        "vision_tower.encoder.0",
    )
    assert config.is_layer_skipped(
        {"modules_to_not_convert": ["lm_head"]},
        "model.lm_head",
    )
    assert patch_humming_regex_ignore() is False


def test_humming_patch_detects_customer_backport(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        def is_layer_skipped(self, config, prefix):
            ignored = self.get_from_keys_or(config, ["ignore"], [])
            return any(
                module_name.startswith("re:")
                and re.match(module_name[3:], prefix)
                for module_name in ignored
            )

    _install_fake_humming(monkeypatch, "0.23.0", HummingConfig)

    assert patch_humming_regex_ignore() is False


def test_humming_patch_skips_when_backend_is_unavailable(monkeypatch):
    class HummingConfig:
        def __init__(self):
            raise AssertionError("Humming backend is unavailable")

    monkeypatch.setattr(
        vllm_patch_version,
        "package_version",
        lambda name: "0.23.0",
    )
    humming = ModuleType("vllm.model_executor.layers.quantization.humming")
    humming.HummingConfig = HummingConfig
    humming.HummingMethod = None
    monkeypatch.setitem(sys.modules, humming.__name__, humming)

    assert patch_humming_regex_ignore() is False


def test_humming_patch_skips_unknown_broken_version(monkeypatch):
    class HummingConfig(_BrokenHummingConfig):
        pass

    _install_fake_humming(monkeypatch, "0.26.0", HummingConfig)

    assert patch_humming_regex_ignore() is False


def test_sglang_entrypoint_configures_logging_before_patches(monkeypatch):
    from modelexpress.engines import sglang as sglang_integration

    calls = []
    monkeypatch.setattr(sglang_integration, "_sglang_registered", False)
    monkeypatch.setattr(
        sglang_integration,
        "configure_sglang_logging",
        lambda: calls.append(("logging", None)),
    )
    monkeypatch.setattr(
        sglang_integration,
        "apply_patches",
        lambda patches: calls.append(("patches", patches)),
    )

    sglang_integration.register_modelexpress_sglang()
    sglang_integration.register_modelexpress_sglang()

    assert calls == [
        ("logging", None),
        ("patches", sglang_integration.SGLANG_PATCHES),
    ]


def _install_fake_sglang_modules(monkeypatch):
    sglang_mod = ModuleType("sglang")
    srt_mod = ModuleType("sglang.srt")
    arg_groups_mod = ModuleType("sglang.srt.arg_groups")
    resolution_hooks_mod = ModuleType("sglang.srt.arg_groups.resolution_hooks")
    overrides_mod = ModuleType("sglang.srt.arg_groups.overrides")
    configs_mod = ModuleType("sglang.srt.configs")
    model_config_mod = ModuleType("sglang.srt.configs.model_config")

    hooks: dict[str, list] = {}

    def register_resolution_hook(step_name: str):
        def decorator(fn):
            hooks.setdefault(step_name, []).append(fn)
            return fn

        return decorator

    def run_hook(step_fn, server_args):
        chain = step_fn
        for hook in hooks.get(step_fn.__name__, ()):
            prev = chain
            chain = lambda args, _h=hook, _p=prev: _h(args, _p)
        return chain(server_args)

    resolution_hooks_mod._HOOKS = hooks
    resolution_hooks_mod.register_resolution_hook = register_resolution_hook
    resolution_hooks_mod.run_hook = run_hook
    overrides_mod.resolving_view = lambda server_args: server_args

    init_calls = []

    class ModelConfig:
        def __init__(self, model_path: str, *args, revision=None, **kwargs):
            init_calls.append((model_path, revision))
            self.model_path = model_path
            self.revision = revision

    model_config_mod.ModelConfig = ModelConfig

    for mod in (
        sglang_mod,
        srt_mod,
        arg_groups_mod,
        resolution_hooks_mod,
        overrides_mod,
        configs_mod,
        model_config_mod,
    ):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)

    return resolution_hooks_mod, ModelConfig, init_calls


def test_sglang_metadata_prefetch_skips_when_disabled(monkeypatch):
    from modelexpress import model_prefetch
    from modelexpress.engines.sglang.patches.patch_sglang_metadata_prefetch import (
        patch_sglang_metadata_prefetch,
    )

    monkeypatch.setattr(model_prefetch, "is_enabled", lambda: False)
    resolution_hooks_mod, ModelConfig, _ = _install_fake_sglang_modules(monkeypatch)
    original_init = ModelConfig.__init__

    assert patch_sglang_metadata_prefetch() is False
    assert resolution_hooks_mod._HOOKS == {}
    assert ModelConfig.__init__ is original_init


def test_sglang_metadata_prefetch_hooks_resolution_and_model_config(monkeypatch):
    from modelexpress import model_prefetch
    from modelexpress.engines.sglang.patches.patch_sglang_metadata_prefetch import (
        patch_sglang_metadata_prefetch,
    )

    monkeypatch.setattr(model_prefetch, "is_enabled", lambda: True)
    prefetched = []
    monkeypatch.setattr(
        model_prefetch,
        "ensure_metadata",
        lambda model, revision=None, *, cache_directory=None: prefetched.append(
            (model, revision, cache_directory)
        ),
    )
    resolution_hooks_mod, ModelConfig, init_calls = _install_fake_sglang_modules(
        monkeypatch
    )

    assert patch_sglang_metadata_prefetch() is True
    assert patch_sglang_metadata_prefetch() is False

    step_order = []

    def handle_model_source_paths(args):
        step_order.append(("resolved", args.model_path))

    server_args = SimpleNamespace(
        model_path="org/model",
        tokenizer_path=None,
        speculative_draft_model_path="org/draft-model",
        revision="main",
        tokenizer_revision="tok-rev",
        speculative_draft_model_revision=None,
        download_dir="/tmp/hf-cache",
    )
    resolution_hooks_mod.run_hook(handle_model_source_paths, server_args)

    assert step_order == [("resolved", "org/model")]
    assert prefetched == [
        ("org/model", "main", "/tmp/hf-cache"),
        ("org/model", "tok-rev", "/tmp/hf-cache"),
        ("org/draft-model", "main", "/tmp/hf-cache"),
    ]

    prefetched.clear()
    cfg = ModelConfig("org/direct-model", revision="v1.0")
    assert cfg.model_path == "org/direct-model"
    assert init_calls == [("org/direct-model", "v1.0")]
    assert prefetched == [("org/direct-model", "v1.0", None)]


def test_sglang_metadata_prefetch_swallows_errors(monkeypatch):
    from modelexpress import model_prefetch
    from modelexpress.engines.sglang.patches.patch_sglang_metadata_prefetch import (
        patch_sglang_metadata_prefetch,
    )

    monkeypatch.setattr(model_prefetch, "is_enabled", lambda: True)

    def broken_ensure(*args, **kwargs):
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(model_prefetch, "ensure_metadata", broken_ensure)
    resolution_hooks_mod, ModelConfig, init_calls = _install_fake_sglang_modules(
        monkeypatch
    )

    assert patch_sglang_metadata_prefetch() is True

    server_args = SimpleNamespace(
        model_path="org/model",
        tokenizer_path=None,
        speculative_draft_model_path=None,
        revision=None,
        tokenizer_revision=None,
        speculative_draft_model_revision=None,
        download_dir=None,
    )
    step_ran = []

    def handle_model_source_paths(args):
        step_ran.append(True)

    resolution_hooks_mod.run_hook(handle_model_source_paths, server_args)
    assert step_ran == [True]

    ModelConfig("org/model", revision="main")
    assert init_calls == [("org/model", "main")]


def test_sglang_object_storage_patch_preserves_modelexpress_load_format(
    monkeypatch,
):
    from modelexpress.engines.sglang.patches.patch_object_storage_format_check import (
        patch_sglang_object_storage_format_check,
    )

    resolution_hooks_mod, _, _ = _install_fake_sglang_modules(monkeypatch)
    overrides_mod = sys.modules["sglang.srt.arg_groups.overrides"]
    declared_overrides = []

    def declare_resolution(server_args, source, **fields):
        declared_overrides.append((source, fields))
        for key, value in fields.items():
            setattr(server_args, key, value)

    overrides_mod.declare_resolution = declare_resolution

    assert patch_sglang_object_storage_format_check() is True
    assert patch_sglang_object_storage_format_check() is False

    def handle_load_format(args):
        if str(args.model_path).startswith("s3://"):
            declare_resolution(
                args, "_handle_load_format", load_format="runai_streamer"
            )
        if getattr(args, "speculative_draft_model_path", None) and str(
            args.speculative_draft_model_path
        ).startswith("s3://"):
            declare_resolution(
                args,
                "_handle_load_format",
                speculative_draft_load_format="runai_streamer",
            )

    server_args = SimpleNamespace(
        model_path="s3://bucket/deepseek-v3",
        speculative_draft_model_path="s3://bucket/deepseek-v3-draft",
        load_format="remote_instance",
        remote_instance_weight_loader_backend="modelexpress",
        speculative_draft_load_format=None,
    )
    resolution_hooks_mod.run_hook(handle_load_format, server_args)

    assert server_args.load_format == "remote_instance"
    assert server_args.speculative_draft_load_format is None
    assert declared_overrides[-1] == (
        "_modelexpress_object_storage_load_format",
        {
            "load_format": "remote_instance",
            "speculative_draft_load_format": None,
        },
    )


def test_sglang_object_storage_patch_ignores_non_modelexpress_load_format(
    monkeypatch,
):
    from modelexpress.engines.sglang.patches.patch_object_storage_format_check import (
        patch_sglang_object_storage_format_check,
    )

    resolution_hooks_mod, _, _ = _install_fake_sglang_modules(monkeypatch)

    assert patch_sglang_object_storage_format_check() is True

    def handle_load_format(args):
        if str(args.model_path).startswith("s3://"):
            args.load_format = "runai_streamer"

    server_args = SimpleNamespace(
        model_path="s3://bucket/deepseek-v3",
        load_format="auto",
        remote_instance_weight_loader_backend="nccl",
        speculative_draft_load_format=None,
    )
    resolution_hooks_mod.run_hook(handle_load_format, server_args)

    assert server_args.load_format == "runai_streamer"
