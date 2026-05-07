from typing import Dict, List, Optional, Tuple, Any
from .graph import GraphNode, ModuleCategory


_MODULE_CATEGORY_PATTERNS: Dict[ModuleCategory, List[str]] = {
    ModuleCategory.PARALLEL_LINEAR: [
        "RowParallelLinear",
        "ColumnParallelLinear",
        "Linear",
        "linear",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "lm_head",
    ],
    ModuleCategory.ATTENTION: [
        "SelfAttention",
        "CrossAttention",
        "attention",
        "Attention",
        "flash_attn",
        "sdpa",
        "ScaledDotProduct",
    ],
    ModuleCategory.MLP: [
        "MLP",
        "FeedForward",
        "mlp",
        "feedforward",
        "swiglu",
        "geglu",
        " gated",
    ],
    ModuleCategory.TRANSFORMER_LAYER: [
        "TransformerLayer",
        "transformer_layer",
        "DecoderLayer",
        "encoder_layer",
        "ParallelTransformerLayer",
    ],
    ModuleCategory.EMBEDDING: [
        "Embedding",
        "embedding",
        "VocabParallelEmbedding",
        "token_embedding",
    ],
    ModuleCategory.NORM: [
        "LayerNorm",
        "RMSNorm",
        "layer_norm",
        "rms_norm",
        "BatchNorm",
        "GroupNorm",
    ],
    ModuleCategory.OPTIMIZER: [
        "Optimizer",
        "optimizer",
        "Adam",
        "SGD",
        "step",
    ],
    ModuleCategory.COMMUNICATION: [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
        "p2p",
    ],
    ModuleCategory.LOSS: [
        "CrossEntropy",
        "MSELoss",
        "loss",
        "Loss",
    ],
}


def infer_module_category(op_type: str, module_path: Optional[str] = None) -> ModuleCategory:
    op_lower = op_type.lower()
    path_lower = (module_path or "").lower()
    text = f"{op_lower} {path_lower}"

    for category, patterns in _MODULE_CATEGORY_PATTERNS.items():
        for pat in patterns:
            if pat.lower() in text:
                return category

    return ModuleCategory.OTHER


def build_hierarchy_paths(nodes: Dict[str, GraphNode]) -> None:
    module_to_node: Dict[str, str] = {}
    for nid, node in nodes.items():
        if node.module_path:
            if node.module_path not in module_to_node:
                module_to_node[node.module_path] = nid
            else:
                existing = nodes[module_to_node[node.module_path]]
                if existing.node_type.value in ("autograd_fwd", "compute"):
                    module_to_node[node.module_path] = nid

    for nid, node in nodes.items():
        node.module_category = infer_module_category(node.op_type, node.module_path).value
        if not node.module_path:
            continue
        parts = node.module_path.split(".")
        node.module_level = len(parts)
        node.hierarchy_path = node.module_path

        for depth in range(len(parts) - 1, 0, -1):
            parent_path = ".".join(parts[:depth])
            if parent_path in module_to_node:
                node.parent_module_id = module_to_node[parent_path]
                break


def get_hierarchy_summary(nodes: Dict[str, GraphNode]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for nid, node in nodes.items():
        cat = node.module_category or "unclassified"
        if cat not in summary:
            summary[cat] = {"count": 0, "node_ids": []}
        summary[cat]["count"] += 1
        summary[cat]["node_ids"].append(nid)
    return summary


def get_nodes_under_hierarchy(nodes: Dict[str, GraphNode], prefix: str) -> List[GraphNode]:
    result: List[GraphNode] = []
    for node in nodes.values():
        path = node.hierarchy_path or node.module_path or ""
        if path.startswith(prefix):
            result.append(node)
    return result
