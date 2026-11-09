#!/usr/bin/env python3
"""Add the first fused attention output as an ONNX graph output for debugging."""

from pathlib import Path
import onnx
from onnx import helper, TensorProto


def main() -> None:
    src = Path("stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-plugin.onnx")
    dst = Path("stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-plugin-debug.onnx")
    model = onnx.load(str(src), load_external_data=True)
    name = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/attention/attention/MatMul_1_output_0"
    model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 12, 1226, 64]))
    prefix = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/attention/attention/"
    model.graph.output.append(helper.make_tensor_value_info(prefix + "Mul_output_0", TensorProto.FLOAT, [1, 12, 1226, 64]))
    model.graph.output.append(helper.make_tensor_value_info(prefix + "Mul_1_output_0", TensorProto.FLOAT, [1, 12, 64, 1226]))
    model.graph.output.append(helper.make_tensor_value_info(prefix + "Transpose_output_0", TensorProto.FLOAT, [1, 12, 1226, 64]))
    onnx.save(model, str(dst), save_as_external_data=False)
    print(dst)


if __name__ == "__main__":
    main()
