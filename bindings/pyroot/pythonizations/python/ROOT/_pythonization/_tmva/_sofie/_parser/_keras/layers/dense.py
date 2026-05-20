def MakeKerasDense(layer):
    """
    Create a Keras-compatible dense (fully connected) layer operation using SOFIE framework.

    This function takes a dictionary representing a dense layer and its attributes and
    constructs a Keras-compatible dense (fully connected) layer operation using the SOFIE framework.
    A dense layer applies a matrix multiplication between the input tensor and weight matrix,
    and adds a bias term.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer weight names, and data type - must be float.

    Returns:
    ROperator_Gemm: A SOFIE framework operator representing the dense layer operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1]
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_transA = 0
    attr_transB = 0
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Gemm["float"](
            attr_alpha, attr_beta, attr_transA, attr_transB, fLayerInputName, fKernelName, fBiasName, fLayerOutputName
        )
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + fLayerDType)
