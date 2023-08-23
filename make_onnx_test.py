from typing import Any, Sequence, Union, List, Optional
import numpy as np
import onnx
from onnx import TypeProto, TensorProto
import onnxruntime
import collections
import os
import shutil
from onnx import numpy_helper

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
    
    
    x = np.random.randn(20, 10, 5).astype(np.float32)
    y = x[0:3, 0:10]
    
    print(x.shape)
    print(y.shape)
    """
    input = ""
    input = "float input[] = {"
    for x_ in np.nditer(x):
        input += str(x_) + ", "
    input = input[:-2]
    input += "};"
    #print(input)

    output = ""
    output = "float output[] = {"
    for y_ in np.nditer(y):
        output += str(y_) + ", "
    output = output[:-2]
    output += "};"
    #print(output)

    """

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [20, 10, 5])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 10, 5])
    starts = onnx.numpy_helper.from_array(np.array([0, 0], dtype = np.int32), name="starts")
    ends = onnx.numpy_helper.from_array(np.array([3, 10], dtype = np.int32), name="ends")
    axes = onnx.numpy_helper.from_array(np.array([0, 1], dtype = np.int32), name="axes")
    steps = onnx.numpy_helper.from_array(np.array([1, 1], dtype = np.int32), name="steps")


    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes", "steps"],
        outputs=["y"],
    )

    graph = onnx.helper.make_graph(
        nodes=[node], name="Slice", inputs=[x], outputs=[y], initializer=[starts, ends, axes, steps]
    )

    model = onnx.helper.make_model(graph, producer_name = "Slice")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print(model)

    onnx.save(model, "Slice.onnx")



