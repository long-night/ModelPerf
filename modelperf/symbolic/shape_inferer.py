from typing import Dict, List, Optional, Tuple, Any, Union
import sympy
from sympy import Symbol, Expr, simplify, symbols


class SymbolicShapeInferer:
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self._init_base_symbols()
        self.shape_rules: Dict[str, callable] = {}
        self._register_default_rules()

    def _init_base_symbols(self):
        symbol_defs = {
            'B': 'batch_size',
            'S': 'sequence_length',
            'H': 'hidden_size',
            'V': 'vocab_size',
            'F': 'ffn_hidden_size',
            'NH': 'num_attention_heads',
            'NG': 'num_query_groups',
            'KV': 'hidden_size_per_attention_head',
            'TP': 'tensor_parallel_size',
            'PP': 'pipeline_parallel_size',
            'DP': 'data_parallel_size',
            'CP': 'context_parallel_size',
            'EP': 'expert_parallel_size',
        }
        for name, desc in symbol_defs.items():
            self.symbols[name] = Symbol(name, positive=True, integer=True)

    def _register_default_rules(self):
        self.shape_rules['ColumnParallelLinear'] = self.column_parallel_linear_shape
        self.shape_rules['RowParallelLinear'] = self.row_parallel_linear_shape
        self.shape_rules['SelfAttention'] = self.self_attention_shape
        self.shape_rules['AttentionOutput'] = self.attention_output_shape
        self.shape_rules['MLP'] = self.mlp_shape
        self.shape_rules['Embedding'] = self.embedding_shape

    def get_symbol(self, name: str) -> Symbol:
        if name not in self.symbols:
            self.symbols[name] = Symbol(name, positive=True, integer=True)
        return self.symbols[name]

    def column_parallel_linear_shape(
        self,
        input_shape: Tuple[Expr, ...],
        in_features: Expr,
        out_features: Expr,
        gather_output: bool = False
    ) -> Tuple[Tuple[Expr, ...], Tuple[Expr, ...], Tuple[Expr, Expr]]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        F_sym = self.get_symbol('F')
        TP_sym = self.get_symbol('TP')

        sharded_out = out_features / TP_sym
        weight_shape = (in_features, sharded_out)

        if gather_output:
            output_shape = (S_sym, B_sym, out_features)
        else:
            output_shape = (S_sym, B_sym, sharded_out)

        return (output_shape, input_shape, weight_shape)

    def row_parallel_linear_shape(
        self,
        input_shape: Tuple[Expr, ...],
        in_features: Optional[Expr] = None,
        out_features: Optional[Expr] = None,
        input_is_parallel: bool = True
    ) -> Tuple[Tuple[Expr, ...], Tuple[Expr, ...], Tuple[Expr, Expr]]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        F_sym = self.get_symbol('F')
        TP_sym = self.get_symbol('TP')

        in_f = in_features if in_features is not None else F_sym
        out_f = out_features if out_features is not None else H_sym

        sharded_in = in_f / TP_sym
        weight_shape = (sharded_in, out_f)

        if input_is_parallel:
            output_shape = (S_sym, B_sym, out_f)
        else:
            output_shape = (S_sym, B_sym, out_f)

        return (output_shape, input_shape, weight_shape)

    def self_attention_shape(
        self,
        hidden_size: Expr,
        num_attention_heads: Expr,
        num_query_groups: Optional[Expr] = None,
        tp_size: Optional[Expr] = None
    ) -> Dict[str, Tuple[Expr, ...]]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        NH_sym = self.get_symbol('NH')
        NG_sym = self.get_symbol('NG')
        KV_sym = self.get_symbol('KV')
        TP_sym = self.get_symbol('TP')

        if tp_size is None:
            tp_size = TP_sym

        head_dim = hidden_size / num_attention_heads

        if num_query_groups is None or num_query_groups == num_attention_heads:
            qkv_output = 3 * hidden_size / tp_size
            weight_shapes = {
                'qkv': (hidden_size, 3 * hidden_size / tp_size)
            }
        else:
            q_per_tp = (num_attention_heads / num_query_groups) * head_dim / tp_size
            kv_per_tp = head_dim / tp_size
            ng_per_tp = num_query_groups / tp_size

            qkv_output = ng_per_tp * (q_per_tp + 2 * kv_per_tp)
            weight_shapes = {
                'q': (hidden_size, num_attention_heads * head_dim / tp_size),
                'k': (hidden_size, num_query_groups * head_dim / tp_size),
                'v': (hidden_size, num_query_groups * head_dim / tp_size)
            }

        shapes = {
            'input': (S_sym, B_sym, hidden_size),
            'qkv_output': (S_sym, B_sym, qkv_output),
            'attn_output': (S_sym, B_sym, hidden_size),
            'weight_shapes': weight_shapes,
            'head_dim': head_dim
        }

        return shapes

    def attention_output_shape(
        self,
        hidden_size: Expr,
        tp_size: Optional[Expr] = None
    ) -> Tuple[Tuple[Expr, ...], Tuple[Expr, ...]]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        TP_sym = self.get_symbol('TP')

        if tp_size is None:
            tp_size = TP_sym

        input_shape = (S_sym, B_sym, hidden_size / tp_size)
        output_shape = (S_sym, B_sym, hidden_size)
        weight_shape = (hidden_size / tp_size, hidden_size)

        return (output_shape, input_shape, weight_shape)

    def mlp_shape(
        self,
        hidden_size: Optional[Expr] = None,
        ffn_hidden_size: Optional[Expr] = None,
        tp_size: Optional[Expr] = None
    ) -> Dict[str, Any]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        F_sym = self.get_symbol('F')
        TP_sym = self.get_symbol('TP')

        h = hidden_size if hidden_size is not None else H_sym
        f = ffn_hidden_size if ffn_hidden_size is not None else F_sym
        tp = tp_size if tp_size is not None else TP_sym

        ffn_sharded = f / tp

        shapes = {
            'input': (S_sym, B_sym, h),
            'fc1_output': (S_sym, B_sym, ffn_sharded),
            'fc2_output': (S_sym, B_sym, h),
            'fc1_weight': (h, ffn_sharded),
            'fc2_weight': (ffn_sharded, h)
        }

        return shapes

    def embedding_shape(
        self,
        vocab_size: Optional[Expr] = None,
        hidden_size: Optional[Expr] = None,
        tp_size: Optional[Expr] = None
    ) -> Dict[str, Any]:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        V_sym = self.get_symbol('V')
        H_sym = self.get_symbol('H')
        TP_sym = self.get_symbol('TP')

        v = vocab_size if vocab_size is not None else V_sym
        h = hidden_size if hidden_size is not None else H_sym
        tp = tp_size if tp_size is not None else TP_sym

        vocab_sharded = v / tp

        shapes = {
            'input_ids': (B_sym, S_sym),
            'output': (S_sym, B_sym, h),
            'weight': (vocab_sharded, h)
        }

        return shapes

    def infer_shape(
        self,
        op_type: str,
        **kwargs
    ) -> Any:
        if op_type not in self.shape_rules:
            raise ValueError(f"Unknown op_type: {op_type}")
        return self.shape_rules[op_type](**kwargs)

    def all_reduce_bytes(
        self,
        tensor_shape: Tuple[Expr, ...],
        dtype_size: int = 2
    ) -> Expr:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')

        numel = 1
        for dim in tensor_shape:
            numel *= dim

        return numel * dtype_size

    def all_gather_bytes(
        self,
        shard_size: Expr,
        world_size: Expr,
        dtype_size: int = 2
    ) -> Expr:
        return shard_size * (world_size - 1) * dtype_size

    def reduce_scatter_bytes(
        self,
        full_size: Expr,
        world_size: Expr,
        dtype_size: int = 2
    ) -> Expr:
        return full_size * (world_size - 1) * dtype_size / world_size

    def all_to_all_bytes(
        self,
        send_size_per_rank: Expr,
        world_size: Expr,
        dtype_size: int = 2
    ) -> Expr:
        return send_size_per_rank * (world_size - 1) * dtype_size

    def tp_all_reduce_bytes(
        self,
        tensor_type: str = 'attention_output'
    ) -> Expr:
        B_sym = self.get_symbol('B')
        S_sym = self.get_symbol('S')
        H_sym = self.get_symbol('H')
        F_sym = self.get_symbol('F')
        TP_sym = self.get_symbol('TP')

        if tensor_type == 'attention_output':
            numel = B_sym * S_sym * H_sym
        elif tensor_type == 'mlp_output':
            numel = B_sym * S_sym * F_sym / TP_sym
        else:
            numel = B_sym * S_sym * H_sym

        return 2 * numel

    def embedding_all_gather_bytes(
        self,
        vocab_size: Optional[Expr] = None,
        hidden_size: Optional[Expr] = None
    ) -> Expr:
        V_sym = self.get_symbol('V')
        H_sym = self.get_symbol('H')
        TP_sym = self.get_symbol('TP')

        if vocab_size is None:
            vocab_size = V_sym
        if hidden_size is None:
            hidden_size = H_sym

        shard_size = vocab_size * hidden_size / TP_sym
        return shard_size * (TP_sym - 1)

    def evaluate(
        self,
        expr: Union[Expr, List[Expr], Dict[str, Expr], Tuple[Expr, ...]],
        values: Dict[str, Union[int, float]]
    ) -> Any:
        sym_values = {self.get_symbol(k) if isinstance(k, str) else k: v for k, v in values.items()}
        if isinstance(expr, (list, tuple)):
            result = []
            for item in expr:
                if isinstance(item, Expr):
                    substituted = item.subs(sym_values)
                    result.append(float(substituted) if substituted.is_number else substituted)
                elif isinstance(item, (list, tuple)):
                    result.append(self.evaluate(item, values))
                else:
                    result.append(item)
            return tuple(result) if isinstance(expr, tuple) else result
        elif isinstance(expr, dict):
            return {k: self.evaluate(v, values) if isinstance(v, (Expr, list, tuple, dict)) else v
                    for k, v in expr.items()}
        elif isinstance(expr, Expr):
            result = expr.subs(sym_values)
            if result.is_number:
                return float(result)
            return result
        else:
            return expr

    def simplify_shape(
        self,
        shape: Tuple[Expr, ...]
    ) -> Tuple[Expr, ...]:
        return tuple(simplify(dim) for dim in shape)

    def compute_tensor_numel(
        self,
        shape: Tuple[Expr, ...]
    ) -> Expr:
        numel = 1
        for dim in shape:
            numel *= dim
        return simplify(numel)

    def compute_memory_bytes(
        self,
        shape: Tuple[Expr, ...],
        dtype_size: int = 2
    ) -> Expr:
        return self.compute_tensor_numel(shape) * dtype_size


def create_inferer_from_configs(
    model_config: Dict[str, Any],
    strategy_config: Dict[str, Any]
) -> SymbolicShapeInferer:
    inferer = SymbolicShapeInferer()

    B = inferer.get_symbol('B')
    S = inferer.get_symbol('S')
    H = inferer.get_symbol('H')
    V = inferer.get_symbol('V')
    F = inferer.get_symbol('F')
    NH = inferer.get_symbol('NH')
    NG = inferer.get_symbol('NG')
    TP = inferer.get_symbol('TP')
    PP = inferer.get_symbol('PP')
    DP = inferer.get_symbol('DP')
    CP = inferer.get_symbol('CP')

    if 'hidden_size' in model_config:
        H_val = model_config['hidden_size']
    if 'num_attention_heads' in model_config:
        NH_val = model_config['num_attention_heads']
    if 'ffn_hidden_size' in model_config:
        F_val = model_config['ffn_hidden_size']

    return inferer


__all__ = [
    'SymbolicShapeInferer',
    'create_inferer_from_configs',
]