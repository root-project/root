from cppyy import gbl as gbl_namespace

def MakeKerasSigmoid(layer):
    """
    Create a Keras-compatible sigmoid activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible sigmoid activation operation using the SOFIE framework.
    Sigmoid is a commonly used activation function that maps input values to the range
    between 0 and 1, providing a way to introduce non-linearity in neural networks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Sigmoid: A SOFIE framework operator representing the sigmoid activation operation.
    """
        
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Sigmoid('float')(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + fLayerDType
        )