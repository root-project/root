from typing import Any, Sequence, Union, List, Optional
import numpy as np
import onnx
from onnx import TypeProto, TensorProto
import onnxruntime
import collections
import os
import shutil
from onnx import numpy_helper

def _batchnorm_test_mode(
    x: np.ndarray,
    s: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    y = s * (x - mean) / np.sqrt(var + epsilon) + bias
    return y.astype(x.dtype)  # type: ignore

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

if __name__=="__main__":
    
    
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    
    input = ""
    input = "std::vector<float> input = {"
    for x_ in np.nditer(x, order = 'C'):
        input += str(x_) + ", "
    input = input[:-2]
    input += "};"
    print(input)

    y = _batchnorm_test_mode(x, s, bias, mean, var)

    output = ""
    output = "float output[] = {"
    for y_ in np.nditer(y, order = 'C'):
        output += str(y_) + ", "
    output = output[:-2]
    output += "};"
    print(output)

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4, 5])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, 4, 5])
    s = onnx.numpy_helper.from_array(s, name="s")
    bias = onnx.numpy_helper.from_array(bias, name="bias")
    mean = onnx.numpy_helper.from_array(mean, name="mean")
    var = onnx.numpy_helper.from_array(var, name="var")
    epsilon = 1e-2


    node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["x", "s", "bias", "mean", "var"],
        outputs=["y"],
        epsilon = epsilon
    )

    graph = onnx.helper.make_graph(
        nodes=[node], name="BatchNormalization", inputs=[x], outputs=[y], initializer=[s, bias, mean, var]
    )

    model = onnx.helper.make_model(graph, producer_name = "BatchNormalization")
    #model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print(model)

    onnx.save(model, "BatchNormalization.onnx")



