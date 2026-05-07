"""Optimizer step capture.

Hooks into torch.optim.Optimizer.step and MegatronOptimizer.step
to mark the optimizer phase, so that AtenCapture can label
optimizer-related ATen ops correctly.
"""
from typing import Optional, Callable, List, Any
import torch
from functools import wraps


class OptimizerCapture:
    """Monitors optimizer step boundaries.

    Usage:
        opt_cap = OptimizerCapture()
        opt_cap.install_hooks()
        # ... training loop ...
        opt_cap.uninstall_hooks()
    """

    _original_torch_step: Optional[Callable] = None
    _original_megatron_step: Optional[Callable] = None
    _instances: List["OptimizerCapture"] = []

    def __init__(self, aten_capture: Optional[Any] = None):
        self._in_optimizer = False
        self.step_count = 0
        self._aten_capture = aten_capture
        OptimizerCapture._instances.append(self)

    @property
    def in_optimizer(self) -> bool:
        return self._in_optimizer

    def _set_optimizer_flag(self, val: bool):
        self._in_optimizer = val
        if self._aten_capture is not None:
            self._aten_capture.set_in_optimizer(val)

    @classmethod
    def install_hooks(cls) -> None:
        cls._install_torch_hook()
        cls._install_megatron_hook()

    @classmethod
    def _install_torch_hook(cls) -> None:
        if cls._original_torch_step is not None:
            return
        orig_step = torch.optim.Optimizer.step
        cls._original_torch_step = orig_step

        @wraps(orig_step)
        def wrapped_step(self, closure=None):
            for inst in cls._instances:
                inst._set_optimizer_flag(True)
            try:
                result = orig_step(self, closure)
                for inst in cls._instances:
                    inst.step_count += 1
                return result
            finally:
                for inst in cls._instances:
                    inst._set_optimizer_flag(False)

        torch.optim.Optimizer.step = wrapped_step

    @classmethod
    def _install_megatron_hook(cls) -> None:
        if cls._original_megatron_step is not None:
            return
        try:
            from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer, FP32Optimizer, ChainedOptimizer
            from megatron.core.optimizer import Float16OptimizerWithFloat16Params
            targets = [
                ('MixedPrecisionOptimizer', MixedPrecisionOptimizer),
                ('Float16OptimizerWithFloat16Params', Float16OptimizerWithFloat16Params),
                ('FP32Optimizer', FP32Optimizer),
                ('ChainedOptimizer', ChainedOptimizer),
            ]
        except ImportError:
            return

        for name, cls_target in targets:
            if not hasattr(cls_target, 'step') or getattr(cls_target, 'step', None) is None:
                continue
            orig_step = cls_target.step
            if cls_target in cls._patched_classes:
                continue

            @wraps(orig_step)
            def make_wrapped(orig):
                def wrapped(self, closure=None):
                    for inst in cls._instances:
                        inst._set_optimizer_flag(True)
                    try:
                        result = orig(self)
                        for inst in cls._instances:
                            inst.step_count += 1
                        return result
                    finally:
                        for inst in cls._instances:
                            inst._set_optimizer_flag(False)
                return wrapped

            cls_target.step = make_wrapped(orig_step)
            cls._patched_classes.add(cls_target)

    _patched_classes = set()

    @classmethod
    def uninstall_hooks(cls) -> None:
        if cls._original_torch_step is not None:
            torch.optim.Optimizer.step = cls._original_torch_step
            cls._original_torch_step = None
        if cls._original_megatron_step is not None:
            try:
                from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
                MixedPrecisionOptimizer.step = cls._original_megatron_step
            except ImportError:
                pass
            cls._original_megatron_step = None

    def reset(self):
        self._in_optimizer = False
        self.step_count = 0
