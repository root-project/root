from cppyy import gbl as gbl_namespace

def MakeKerasTanh(layer):
    """
    Create a Keras-compatible hyperbolic tangent (tanh) activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible tanh activation operation using the SOFIE framework.
    Tanh is an activation function that squashes input values to the range between -1 and 1,
    introducing non-linearity in neural networks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Tanh: A SOFIE framework operator representing the tanh activation operation.
    """
        
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Tanh('float')(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Tanh does not yet support input type " + fLayerDType
        )