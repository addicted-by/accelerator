import pathlib
from typing import AnyStr, Optional, Union

import onnx


def _get_input_output_names(onnx_model: onnx.onnx_ml_pb2.ModelProto):
    output = [node.name for node in onnx_model.graph.output]

    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    return output, net_feed_input


def get_input_output_names(
    onnx_model: Union[pathlib.Path, AnyStr, onnx.onnx_ml_pb2.ModelProto],
    return_outputs=True,
    return_inputs=False,
    return_model=False,
) -> tuple[Optional[list], Optional[list], Optional[onnx.onnx_ml_pb2.ModelProto]]:
    if isinstance(onnx_model, (pathlib.Path, str)):
        print(f"Loading model from {onnx_model}")
        onnx_model = onnx.load(onnx_model)

    output, net_input = _get_input_output_names(onnx_model)

    result = tuple()

    if return_outputs:
        result += (output,)

    if return_inputs:
        result += (net_input,)

    if return_model:
        result += (return_model,)

    return result
