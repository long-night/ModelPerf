import unittest
from modelperf.symbolic.shape_inferer import SymbolicShapeInferer


class TestSymbolicShapeInferer(unittest.TestCase):
    def test_basic_initialization(self):
        inferer = SymbolicShapeInferer()
        self.assertIsNotNone(inferer.symbols)
        self.assertIn("B", inferer.symbols)
        self.assertIn("S", inferer.symbols)
        self.assertIn("H", inferer.symbols)
        
    def test_base_symbols_exist(self):
        inferer = SymbolicShapeInferer()
        expected_symbols = [
            "B", "S", "H", "V", "F", "NH", "NG", "KV",
            "TP", "PP", "DP", "CP", "EP"
        ]
        for sym in expected_symbols:
            self.assertIn(sym, inferer.symbols)
            
    def test_get_symbol_existing(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        self.assertIsNotNone(B)
        self.assertEqual(str(B), "B")
        
    def test_get_symbol_new(self):
        inferer = SymbolicShapeInferer()
        X = inferer.get_symbol("X")
        self.assertIn("X", inferer.symbols)
        self.assertEqual(str(X), "X")
        
    def test_column_parallel_linear_shape(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        S = inferer.get_symbol("S")
        H = inferer.get_symbol("H")
        F = inferer.get_symbol("F")
        TP = inferer.get_symbol("TP")
        
        output_shape, input_shape, weight_shape = inferer.column_parallel_linear_shape(
            input_shape=(S, B, H),
            in_features=H,
            out_features=F,
            gather_output=False
        )
        
        self.assertEqual(output_shape[0], S)
        self.assertEqual(output_shape[1], B)
        self.assertEqual(input_shape[0], S)
        self.assertEqual(input_shape[1], B)
        
    def test_column_parallel_linear_gather_output(self):
        inferer = SymbolicShapeInferer()
        F = inferer.get_symbol("F")
        
        output_shape, input_shape, weight_shape = inferer.column_parallel_linear_shape(
            input_shape=(None, None, None),
            in_features=None,
            out_features=F,
            gather_output=True
        )
        
        self.assertEqual(output_shape[2], F)
        
    def test_row_parallel_linear_shape(self):
        inferer = SymbolicShapeInferer()
        output_shape, input_shape, weight_shape = inferer.row_parallel_linear_shape(
            input_shape=(None, None, None),
            in_features=None,
            out_features=None,
            input_is_parallel=True
        )
        self.assertIsNotNone(output_shape)
        
    def test_self_attention_shape(self):
        inferer = SymbolicShapeInferer()
        H = inferer.get_symbol("H")
        NH = inferer.get_symbol("NH")
        
        shapes = inferer.self_attention_shape(
            hidden_size=H,
            num_attention_heads=NH,
            num_query_groups=None,
            tp_size=None
        )
        
        self.assertIn("input", shapes)
        self.assertIn("qkv_output", shapes)
        self.assertIn("attn_output", shapes)
        self.assertIn("weight_shapes", shapes)
        
    def test_attention_output_shape(self):
        inferer = SymbolicShapeInferer()
        H = inferer.get_symbol("H")
        TP = inferer.get_symbol("TP")
        
        output_shape, input_shape, weight_shape = inferer.attention_output_shape(
            hidden_size=H,
            tp_size=TP
        )
        
        self.assertEqual(input_shape[2], H / TP)
        self.assertEqual(output_shape[2], H)
        
    def test_mlp_shape(self):
        inferer = SymbolicShapeInferer()
        shapes = inferer.mlp_shape(
            hidden_size=None,
            ffn_hidden_size=None,
            tp_size=None
        )
        
        self.assertIn("input", shapes)
        self.assertIn("fc1_output", shapes)
        self.assertIn("fc2_output", shapes)
        self.assertIn("fc1_weight", shapes)
        self.assertIn("fc2_weight", shapes)
        
    def test_embedding_shape(self):
        inferer = SymbolicShapeInferer()
        shapes = inferer.embedding_shape(
            vocab_size=None,
            hidden_size=None,
            tp_size=None
        )
        
        self.assertIn("input_ids", shapes)
        self.assertIn("output", shapes)
        self.assertIn("weight", shapes)
        
    def test_compute_tensor_numel(self):
        inferer = SymbolicShapeInferer()
        numel = inferer.compute_tensor_numel((4, 8, 16))
        self.assertEqual(numel, 4 * 8 * 16)
        
    def test_compute_memory_bytes(self):
        inferer = SymbolicShapeInferer()
        bytes_count = inferer.compute_memory_bytes((4, 8), dtype_size=2)
        self.assertEqual(bytes_count, 4 * 8 * 2)
        
    def test_compute_memory_bytes_fp32(self):
        inferer = SymbolicShapeInferer()
        bytes_count = inferer.compute_memory_bytes((4, 8), dtype_size=4)
        self.assertEqual(bytes_count, 4 * 8 * 4)
        
    def test_evaluate_simple(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        result = inferer.evaluate(B, {"B": 8})
        self.assertEqual(result, 8.0)
        
    def test_evaluate_expression(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        S = inferer.get_symbol("S")
        expr = B * S
        result = inferer.evaluate(expr, {"B": 8, "S": 128})
        self.assertEqual(result, 8.0 * 128.0)
        
    def test_evaluate_list(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        result = inferer.evaluate([B, B * 2], {"B": 8})
        self.assertEqual(result, [8.0, 16.0])
        
    def test_evaluate_dict(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        result = inferer.evaluate({"a": B, "b": B * 2}, {"B": 8})
        self.assertEqual(result, {"a": 8.0, "b": 16.0})
        
    def test_simplify_shape(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        S = inferer.get_symbol("S")
        shape = (B * 1, S + 0, B * S / B)
        simplified = inferer.simplify_shape(shape)
        self.assertEqual(simplified[0], B)
        
    def test_all_reduce_bytes(self):
        inferer = SymbolicShapeInferer()
        tensor_shape = (4, 8, 16)
        bytes_count = inferer.all_reduce_bytes(tensor_shape, dtype_size=2)
        expected = 4 * 8 * 16 * 2
        self.assertEqual(bytes_count, expected)
        
    def test_all_gather_bytes(self):
        inferer = SymbolicShapeInferer()
        TP = inferer.get_symbol("TP")
        shard_size = 1024
        result = inferer.all_gather_bytes(shard_size, TP, dtype_size=2)
        self.assertEqual(result, 1024 * (TP - 1) * 2)
        
    def test_reduce_scatter_bytes(self):
        inferer = SymbolicShapeInferer()
        TP = inferer.get_symbol("TP")
        full_size = 1024 * 8
        result = inferer.reduce_scatter_bytes(full_size, TP, dtype_size=2)
        self.assertEqual(result, full_size * (TP - 1) * 2 / TP)
        
    def test_all_to_all_bytes(self):
        inferer = SymbolicShapeInferer()
        TP = inferer.get_symbol("TP")
        send_size = 1024
        result = inferer.all_to_all_bytes(send_size, TP, dtype_size=2)
        self.assertEqual(result, send_size * (TP - 1) * 2)
        
    def test_tp_all_reduce_bytes_attention(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        S = inferer.get_symbol("S")
        H = inferer.get_symbol("H")
        result = inferer.tp_all_reduce_bytes(tensor_type="attention_output")
        expected = 2 * B * S * H
        self.assertEqual(result, expected)
        
    def test_tp_all_reduce_bytes_mlp(self):
        inferer = SymbolicShapeInferer()
        B = inferer.get_symbol("B")
        S = inferer.get_symbol("S")
        F = inferer.get_symbol("F")
        TP = inferer.get_symbol("TP")
        result = inferer.tp_all_reduce_bytes(tensor_type="mlp_output")
        expected = 2 * B * S * F / TP
        self.assertEqual(result, expected)
        
    def test_embedding_all_gather_bytes(self):
        inferer = SymbolicShapeInferer()
        V = inferer.get_symbol("V")
        H = inferer.get_symbol("H")
        TP = inferer.get_symbol("TP")
        result = inferer.embedding_all_gather_bytes(V, H)
        expected = V * H / TP * (TP - 1)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
