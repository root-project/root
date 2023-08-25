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
    
    
    shape = (2, 3)
    data = np.random.random_sample(shape).astype(np.float32)
    
    input = ""
    input = "std::vector<float> input = {"
    for x_ in np.nditer(data, order = 'C'):
        input += str(x_) + ", "
    input = input[:-2]
    input += "};"
    print(input)
    

    transposed = np.transpose(data)

    output = ""
    output = "float output[] = {"
    for y_ in np.nditer(transposed, order = 'C'):
        output += str(y_) + ", "
    output = output[:-2]
    output += "};"
    print(output)

    print(data.shape)
    print(transposed.shape)
    # data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [1, 1, 3, 4, 5])
    # tranposed = onnx.helper.make_tensor_value_info("tranposed", onnx.TensorProto.FLOAT, [1, 2, 5, 6, 7])
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    transposed = onnx.helper.make_tensor_value_info("transposed", onnx.TensorProto.FLOAT, [3, 2])
    # perm = onnx.numpy_helper.from_array(np.array([2, 0, 1]).astype(np.int32), name="perm")
    #data = onnx.numpy_helper.from_array(data, name="data")
    #transposed = onnx.numpy_helper.from_array(transposed, name="transposed")
    #W = onnx.numpy_helper.from_array(w, name="W")


    node = onnx.helper.make_node(
        "Transpose",
        inputs=["data"],
        outputs=["transposed"],
        perm=perm
    )

    graph = onnx.helper.make_graph(
        nodes=[node], name="Transpose_permutation", inputs=[data], outputs=[transposed]
    )

    model = onnx.helper.make_model(graph, producer_name = "Transpose_permutation")
    #model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    print(model)

    onnx.save(model, "Transpose_permutation.onnx")



