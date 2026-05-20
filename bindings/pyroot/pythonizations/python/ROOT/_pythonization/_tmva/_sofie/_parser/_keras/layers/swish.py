def MakeKerasSwish(layer):
    """
    Create a Keras-compatible swish activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible swish activation operation using the SOFIE framework.
    Swish is an activation function that aims to combine the benefits of ReLU and sigmoid,
    allowing some non-linearity while still keeping positive values unbounded.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type.

    Returns:
    ROperator_Swish: A SOFIE framework operator representing the swish activation operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Swish("float")(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Swish does not yet support input type " + fLayerDType)
