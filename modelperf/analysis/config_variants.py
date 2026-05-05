from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum, auto


class ConfigChangeType(Enum):
    PARALLEL_STRATEGY = auto()
    MODEL_CONFIG = auto()
    BATCH_CONFIG = auto()
    SYSTEM_CONFIG = auto()


@dataclass
class ConfigVariant:
    name: str
    changes: Dict[str, Any] = field(default_factory=dict)
    change_type: ConfigChangeType = ConfigChangeType.PARALLEL_STRATEGY
    description: str = ""
    constraints: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)

    def validate(self, base_config: Dict[str, Any]) -> bool:
        for key, constraint in self.constraints.items():
            value = self.changes.get(key, base_config.get(key))
            if value is not None and not constraint(value):
                return False
        return True

    def apply(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate(base_config):
            raise ValueError(f"Config variant '{self.name}' failed validation")
        result = dict(base_config)
        result.update(self.changes)
        return result


@dataclass
class ConfigSweep:
    name: str
    param_name: str
    values: List[Any]
    change_type: ConfigChangeType = ConfigChangeType.PARALLEL_STRATEGY

    def generate_variants(self) -> List[ConfigVariant]:
        variants = []
        for val in self.values:
            name = f"{self.name}_{self.param_name}{val}"
            variant = ConfigVariant(
                name=name,
                changes={self.param_name: val},
                change_type=self.change_type
            )
            variants.append(variant)
        return variants


def create_tp_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="tp",
        param_name="tensor_parallel_size",
        values=values,
        change_type=ConfigChangeType.PARALLEL_STRATEGY
    )


def create_pp_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="pp",
        param_name="pipeline_parallel_size",
        values=values,
        change_type=ConfigChangeType.PARALLEL_STRATEGY
    )


def create_dp_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="dp",
        param_name="data_parallel_size",
        values=values,
        change_type=ConfigChangeType.PARALLEL_STRATEGY
    )


def create_cp_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="cp",
        param_name="context_parallel_size",
        values=values,
        change_type=ConfigChangeType.PARALLEL_STRATEGY
    )


def create_ep_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="ep",
        param_name="expert_parallel_size",
        values=values,
        change_type=ConfigChangeType.PARALLEL_STRATEGY
    )


def create_batch_size_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="bs",
        param_name="micro_batch_size",
        values=values,
        change_type=ConfigChangeType.BATCH_CONFIG
    )


def create_hidden_size_sweep(values: List[int]) -> ConfigSweep:
    return ConfigSweep(
        name="hs",
        param_name="hidden_size",
        values=values,
        change_type=ConfigChangeType.MODEL_CONFIG
    )


def generate_grid_search(sweeps: List[ConfigSweep]) -> List[ConfigVariant]:
    from itertools import product

    all_variants = [sweep.generate_variants() for sweep in sweeps]

    combinations = []
    for combo in product(*all_variants):
        merged_changes = {}
        for variant in combo:
            merged_changes.update(variant.changes)

        name_parts = [v.name for v in combo]
        name = "_".join(name_parts)

        combined = ConfigVariant(
            name=name,
            changes=merged_changes,
            change_type=ConfigChangeType.PARALLEL_STRATEGY
        )
        combinations.append(combined)

    return combinations


__all__ = [
    'ConfigChangeType',
    'ConfigVariant',
    'ConfigSweep',
    'create_tp_sweep',
    'create_pp_sweep',
    'create_dp_sweep',
    'create_cp_sweep',
    'create_ep_sweep',
    'create_batch_size_sweep',
    'create_hidden_size_sweep',
    'generate_grid_search',
]
