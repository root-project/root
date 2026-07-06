import os
import time


def move_operator(op):
    """
    Wrap an operator into a std::unique_ptr to pass it to RModel::AddOperator().
    """
    import ROOT

    # If the object is already held by a smart pointer, just move it.
    smartptr = op.__smartptr__()
    if smartptr:
        return type(smartptr)(ROOT.std.move(smartptr))

    ROOT.SetOwnership(op, False)
    return ROOT.std.unique_ptr[type(op)](op)


def _node_get(node, key):
    """
    Get the value of an attribute of a PyTorch ONNX Graph node.

    The helper function is used to avoid dependency on the onnx submodule (for the
    subscript operator of torch._C.Node), as done in https://github.com/pytorch/pytorch/pull/82628
    """
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


def MakePyTorchGemm(node):
    """
    Create a SOFIE Gemm operator from a PyTorch ONNX Graph node.

    For PyTorch's Linear layer having Gemm operation in its ONNX graph,
    the names of the input tensor, output tensor are extracted, and then
    are passed to instantiate a ROperator_Gemm object using the required attributes.
    The node inputs are a list of tensor names, which includes the names of the
    input tensor and the weight tensors.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes,
                 input & output tensor names and the data type (must be float).

    Returns:
    ROperator_Gemm: A SOFIE framework operator representing the Gemm operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    attributes = node["nodeAttributes"]
    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]
    node_dtype = node["nodeDType"][0]

    # Extracting the parameters for the Gemm operator
    name_a = inputs[0]
    name_b = inputs[1]
    name_c = inputs[2]
    name_y = outputs[0]
    attr_alpha = float(attributes["alpha"])
    attr_beta = float(attributes["beta"])

    if "transB" in attributes:
        attr_transB = int(attributes["transB"])
        attr_transA = int(not attr_transB)
    else:
        attr_transA = int(attributes["transA"])
        attr_transB = int(not attr_transA)

    if SOFIE.ConvertStringToType(node_dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Gemm["float"](
            attr_alpha, attr_beta, attr_transA, attr_transB, name_a, name_b, name_c, name_y
        )
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + node_dtype)


def MakePyTorchConv(node):
    """
    Create a SOFIE Conv operator from a PyTorch ONNX Graph node.

    For the Conv operator of PyTorch's ONNX Graph, attributes like dilations, group,
    kernel shape, pads and strides are found, and are passed in instantiating the
    ROperator object with autopad default to `NOTSET`.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes,
                 input & output tensor names and the data type (must be float).

    Returns:
    ROperator_Conv: A SOFIE framework operator representing the Conv operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    attributes = node["nodeAttributes"]
    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]
    node_dtype = node["nodeDType"][0]

    # Extracting the Conv node attributes
    attr_autopad = "NOTSET"
    attr_dilations = list(attributes["dilations"])
    attr_group = int(attributes["group"])
    attr_kernel_shape = list(attributes["kernel_shape"])
    attr_pads = list(attributes["pads"])
    attr_strides = list(attributes["strides"])
    name_x = inputs[0]
    name_w = inputs[1]
    name_b = inputs[2]
    name_y = outputs[0]

    if SOFIE.ConvertStringToType(node_dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Conv["float"](
            attr_autopad,
            attr_dilations,
            attr_group,
            attr_kernel_shape,
            attr_pads,
            attr_strides,
            name_x,
            name_w,
            name_b,
            name_y,
        )
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Conv does not yet support input type " + node_dtype)


def MakePyTorchRelu(node):
    """
    Create a SOFIE Relu operator from a PyTorch ONNX Graph node.

    For instantiating a ROperator_Relu object, the names of
    input & output tensors and the data type of the node are extracted.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes,
                 input & output tensor names and the data type (must be float).

    Returns:
    ROperator_Relu: A SOFIE framework operator representing the Relu operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]
    node_dtype = node["nodeDType"][0]

    if SOFIE.ConvertStringToType(node_dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Relu["float"](inputs[0], outputs[0])
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + node_dtype)


def MakePyTorchSelu(node):
    """
    Create a SOFIE Selu operator from a PyTorch ONNX Graph node.

    For instantiating a ROperator_Selu object, the names of
    input & output tensors and the data type of the node are extracted.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes,
                 input & output tensor names and the data type (must be float).

    Returns:
    ROperator_Selu: A SOFIE framework operator representing the Selu operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]
    node_dtype = node["nodeDType"][0]

    if SOFIE.ConvertStringToType(node_dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Selu["float"](inputs[0], outputs[0])
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Selu does not yet support input type " + node_dtype)


def MakePyTorchSigmoid(node):
    """
    Create a SOFIE Sigmoid operator from a PyTorch ONNX Graph node.

    For instantiating a ROperator_Sigmoid object, the names of
    input & output tensors and the data type of the node are extracted.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes,
                 input & output tensor names and the data type (must be float).

    Returns:
    ROperator_Sigmoid: A SOFIE framework operator representing the Sigmoid operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]
    node_dtype = node["nodeDType"][0]

    if SOFIE.ConvertStringToType(node_dtype) == SOFIE.ETensorType.FLOAT:
        return SOFIE.ROperator_Sigmoid["float"](inputs[0], outputs[0])
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + node_dtype)


def MakePyTorchTranspose(node):
    """
    Create a SOFIE Transpose operator from a PyTorch ONNX Graph node.

    For the Transpose operator of PyTorch's ONNX Graph, the permute dimensions are
    found, and are passed in instantiating the ROperator object.

    Parameters:
    node (dict): A dictionary containing node information including type, attributes
                 and input & output tensor names.

    Returns:
    ROperator_Transpose: A SOFIE framework operator representing the Transpose operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    attributes = node["nodeAttributes"]
    inputs = node["nodeInputs"]
    outputs = node["nodeOutputs"]

    # Extracting the permute dimensions for the transpose
    attr_perm = [int(dim) for dim in attributes["perm"]]

    return SOFIE.ROperator_Transpose(attr_perm, inputs[0], outputs[0])


# Set global dictionary, mapping PyTorch ONNX Graph nodes to corresponding functions
# that create their ROperator instances
mapPyTorchNode = {
    "onnx::Gemm": MakePyTorchGemm,
    "onnx::Conv": MakePyTorchConv,
    "onnx::Relu": MakePyTorchRelu,
    "onnx::Selu": MakePyTorchSelu,
    "onnx::Sigmoid": MakePyTorchSigmoid,
    "onnx::Transpose": MakePyTorchTranspose,
}


def MakePyTorchNode(node):
    """
    Prepare the equivalent ROperator with respect to a PyTorch ONNX Graph node.

    The function searches for the passed PyTorch ONNX Graph node in the map, and calls
    the specific preparatory function, subsequently returning the ROperator object.

    For developing new preparatory functions for supporting PyTorch ONNX Graph nodes
    in the future, all one needs is to extract the required properties and attributes
    from the node dictionary, which contains all the information about any PyTorch ONNX
    Graph node, and after any required transformations, these are passed for instantiating
    the ROperator object.

    The node dictionary which holds all the information about a PyTorch ONNX Graph's node
    has the following structure:

        dict node {  'nodeType'        : Type of node (operator)
                     'nodeAttributes'  : Attributes of the node
                     'nodeInputs'      : List of names of input tensors
                     'nodeOutputs'     : List of names of output tensors
                     'nodeDType'       : Data type of the operator node
                  }

    Parameters:
    node (dict): A dictionary representing a PyTorch ONNX Graph node.

    Returns:
    ROperator: A SOFIE framework operator representing the node operation.
    """
    node_type = node["nodeType"]
    if node_type not in mapPyTorchNode:
        raise RuntimeError("TMVA::SOFIE - Parsing PyTorch node " + node_type + " is not yet supported")
    return mapPyTorchNode[node_type](node)


class PyTorch:
    def Parse(filename, input_shapes, input_dtypes=None):
        """
        Parse a trained PyTorch .pt model into a RModel object.

        The parser uses internal functions of PyTorch to convert any PyTorch model
        into its equivalent ONNX Graph. For this conversion, dummy inputs are built
        which are passed through the model and the applied operators are recorded
        for populating the ONNX graph. This requires the shapes and data types of
        the input tensors, which are used for building the dummy inputs.
        After the said conversion, the nodes of the ONNX graph are then traversed to
        extract properties like node type, attributes and input & output tensor names,
        and the equivalent ROperator instances are added into the RModel object.

        The internal function used to convert the model to a graph object returns a
        list which contains a Graph object and a dictionary of weights. This dictionary
        is used to add the initialized tensors of the model into the RModel object.

        For adding the input tensor infos, the names of the input tensors are extracted
        from the PyTorch ONNX graph object. The vectors of shapes & data types passed
        into the Parse() function are used for the shapes and the data types of the
        input tensors.

        For the output tensor infos, the names of the output tensors are also extracted
        from the Graph object and are then added into the RModel object.

        Parameters:
        filename (str): File location of the PyTorch .pt model.
        input_shapes: List of input shape lists.
        input_dtypes: Optional list of SOFIE.ETensorType for the data types of the
                      input tensors. Defaults to float for all input tensors.

        Returns:
        RModel: The parsed model.

        Example usage:
        ~~~ {.py}
        import ROOT
        model = ROOT.TMVA.Experimental.SOFIE.PyTorch.Parse("trained_model_dense.pt", [[120, 1]])
        ~~~
        """

        # PyTorch is too fragile to import unconditionally. As its presence might break several ROOT
        # usecases and importing torch globally will slow down importing ROOT, which is not desired.
        # For this, we import torch within the functions instead of importing it at the start of the
        # file (i.e. globally). So, whenever the parser function is called, only then torch will be
        # imported, and not everytime we import ROOT. Also, we can import torch in multiple functions
        # as many times as we want since Python caches the imported packages.

        import torch
        from ROOT.TMVA.Experimental import SOFIE
        from torch.onnx.utils import _model_to_graph

        # Check if file exists
        if not os.path.exists(filename):
            raise RuntimeError("Model file {} not found!".format(filename))

        # create new RModel object
        sep = "/"
        if os.name == "nt":
            sep = "\\"

        isep = filename.rfind(sep)
        filename_nodir = filename
        if isep != -1:
            filename_nodir = filename[isep + 1 :]

        ttime = time.time()
        gmt_time = time.gmtime(ttime)
        parsetime = time.asctime(gmt_time)

        rmodel = SOFIE.RModel(filename_nodir, parsetime)

        print("Torch Version: " + torch.__version__)

        # The data types of the input tensors default to float
        if input_dtypes is None:
            input_dtypes = [SOFIE.ETensorType.FLOAT] * len(input_shapes)

        # Load the model and prepare it for the conversion to an ONNX graph
        model = torch.jit.load(filename)
        model.cpu()
        model.eval()

        # Build dummy inputs for the model from the given input shapes
        dummy_inputs = [torch.rand(*[int(dim) for dim in shape]) for shape in input_shapes]

        # Get the ONNX graph from the model using the dummy inputs
        graph = _model_to_graph(model, dummy_inputs)

        # Iterate over the nodes of the ONNX graph and add the equivalent operators to the RModel
        for node in graph[0].nodes():
            node_data = {}
            node_data["nodeType"] = node.kind()
            node_data["nodeAttributes"] = {name: _node_get(node, name) for name in node.attributeNames()}
            node_data["nodeInputs"] = [x.debugName() for x in node.inputs()]
            node_data["nodeOutputs"] = [x.debugName() for x in node.outputs()]
            node_data["nodeDType"] = [x.type().scalarType() for x in node.outputs()]

            # Adding required routines depending on the node types for generating inference code
            node_type = node_data["nodeType"]
            if node_type == "onnx::Gemm":
                rmodel.AddBlasRoutines(["Gemm", "Gemv"])
            elif node_type == "onnx::Selu" or node_type == "onnx::Sigmoid":
                rmodel.AddNeededStdLib("cmath")
            elif node_type == "onnx::Conv":
                rmodel.AddBlasRoutines(["Gemm", "Axpy"])

            rmodel.AddOperator(move_operator(MakePyTorchNode(node_data)))

        # Extract the model weights to add the initialized tensors to the RModel
        for weight_name, weight_tensor in graph[1].items():
            # e.g. "torch.FloatTensor" -> "Float"
            weight_dtype = SOFIE.ConvertStringToType(weight_tensor.type()[6:-6])
            if weight_dtype == SOFIE.ETensorType.FLOAT:
                weight_value = weight_tensor.numpy()
                rmodel.AddInitializedTensor(
                    weight_name, SOFIE.ETensorType.FLOAT, list(weight_value.shape), weight_value.flatten()
                )
            else:
                raise RuntimeError(
                    "Type error: TMVA SOFIE does not yet supports weights of data type "
                    + SOFIE.ConvertTypeToString(weight_dtype)
                )

        # Extract the input tensor infos (the first graph input is the model itself)
        input_names = [x.debugName() for x in model.graph.inputs()][1:]
        for input_name, input_shape, input_dtype in zip(input_names, input_shapes, input_dtypes):
            if input_dtype == SOFIE.ETensorType.FLOAT:
                rmodel.AddInputTensorInfo(input_name, SOFIE.ETensorType.FLOAT, [int(dim) for dim in input_shape])
                rmodel.AddInputTensorName(input_name)
            else:
                raise RuntimeError(
                    "Type Error: TMVA SOFIE does not yet support the input tensor data type "
                    + SOFIE.ConvertTypeToString(input_dtype)
                )

        # Extract the output tensor names
        output_names = [x.debugName() for x in graph[0].outputs()]
        rmodel.AddOutputTensorNameList(output_names)

        return rmodel
