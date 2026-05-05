"""ModelPerf framework adapter for Megatron-LM and Pai-Megatron-Patch.

通过 Monkey-patch 在训练启动时自动提取模型配置、并行策略和运行时参数，
实现零配置仿真。完全基于 CPU 环境运行。
"""

from .megatron_hooks import (
    MegatronHookManager,
    register_megatron_hooks,
    unregister_megatron_hooks,
    get_captured_configs,
)
from .config_extractor import ConfigExtractor, extract_configs_from_args
from .pai_patch_hooks import PaiPatchHookManager, register_pai_patch_hooks

__all__ = [
    'MegatronHookManager',
    'register_megatron_hooks',
    'unregister_megatron_hooks',
    'get_captured_configs',
    'ConfigExtractor',
    'extract_configs_from_args',
    'PaiPatchHookManager',
    'register_pai_patch_hooks',
]
