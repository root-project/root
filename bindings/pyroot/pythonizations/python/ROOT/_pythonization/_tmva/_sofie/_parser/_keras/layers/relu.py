from cppyy import gbl as gbl_namespace

def MakeKerasReLU(layer):
    """
    Create a Keras-compatible rectified linear unit (ReLU) activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible ReLU activation operation using the SOFIE framework.
    ReLU is a popular activation function that replaces all negative values in a tensor
    with zero, while leaving positive values unchanged.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type, which must be float.

    Returns:
    ROperator_Relu: A SOFIE framework operator representing the ReLU activation operation.
    """
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Relu('float')(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + fLayerDType
        )