def MakeKerasPermute(layer):
    """
    Create a Keras-compatible permutation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible permutation operation using the SOFIE framework.
    Permutation is an operation that rearranges the dimensions of a tensor based on
    specified dimensions.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  attributes, and data type - must be float.

    Returns:
    ROperator_Transpose: A SOFIE framework operator representing the permutation operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fAttributePermute = list(attributes["dims"])
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        if len(fAttributePermute) > 0:
            fAttributePermute = [0] + fAttributePermute  # for the batch dimension from the input
            op = SOFIE.ROperator_Transpose("float")(
                fAttributePermute, fLayerInputName, fLayerOutputName
            )  # SOFIE.fPermuteDims
        else:
            op = SOFIE.ROperator_Transpose("float")(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + fLayerDType
        )
