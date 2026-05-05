"""Pai-Megatron-Patch specific hooks and adapters."""

from typing import Any, Dict, Optional
from functools import wraps

from .megatron_hooks import MegatronHookManager


class PaiPatchHookManager(MegatronHookManager):
    def __init__(self, export_dir: Optional[str] = None):
        super().__init__(export_dir=export_dir)
        self._pai_patch_args: Optional[Dict] = None
    
    def install(self) -> None:
        super().install()
        self._install_pai_patch_args_hook()
    
    def _install_pai_patch_args_hook(self) -> None:
        try:
            from megatron_patch.arguments import get_patch_args
        except ImportError:
            print("[ModelPerf] Warning: megatron_patch.arguments not found, skipping Pai-Patch hook")
            return
        
        original_func = get_patch_args
        
        @wraps(original_func)
        def hooked_get_patch_args(parser):
            parser = original_func(parser)
            
            try:
                from megatron.training import get_args
                args = get_args()
                
                pai_fields = [
                    'use_legacy_models', 'swiglu', 'use_cpu_initialization',
                    'rotary_interleaved', 'dataset', 'train_data_path',
                    'valid_data_path', 'test_data_path', 'data_cache_path',
                    'split', 'tokenizer_type', 'tokenizer_model',
                ]
                
                self._pai_patch_args = {}
                for field in pai_fields:
                    val = getattr(args, field, None)
                    if val is not None:
                        self._pai_patch_args[field] = val
                
                print(f"[ModelPerf] Captured Pai-Patch args: {len(self._pai_patch_args)} fields")
                
            except Exception as e:
                print(f"[ModelPerf] Warning: Failed to capture Pai-Patch args: {e}")
            
            return parser
        
        try:
            import megatron_patch.arguments as patch_args_module
            patch_args_module.get_patch_args = hooked_get_patch_args
        except ImportError:
            pass
    
    def get_pai_patch_args(self) -> Optional[Dict]:
        return self._pai_patch_args


def register_pai_patch_hooks(export_dir: Optional[str] = None) -> PaiPatchHookManager:
    manager = PaiPatchHookManager(export_dir=export_dir)
    manager.install()
    return manager
