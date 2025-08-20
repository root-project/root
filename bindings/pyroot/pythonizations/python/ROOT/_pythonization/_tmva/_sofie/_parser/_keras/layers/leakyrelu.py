from cppyy import gbl as gbl_namespace

def MakeKerasLeakyRelu(layer):
    """
    Create a Keras-compatible Leaky ReLU activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible Leaky ReLU activation operation using the SOFIE framework.
    Leaky ReLU is a variation of the ReLU activation function that allows small negative
    values to pass through, introducing non-linearity while preventing "dying" neurons.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  attributes, and data type - must be float.

    Returns:
    ROperator_LeakyRelu: A SOFIE framework operator representing the Leaky ReLU activation operation.
    """
        
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer['layerAttributes']
    
    if 'alpha' in attributes.keys():
        fAlpha = float(attributes["alpha"])
    elif 'activation' in attributes.keys():
        fAlpha = float(attributes['activation'].alpha)
    elif 'negative_slope' in attributes.keys():
        fAlpha = float(attributes['negative_slope'])
    else:
        raise RuntimeError (
            "Failed to extract alpha value from LeakyReLU"
        )
        
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_LeakyRelu('float')(fAlpha, fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator LeakyRelu does not yet support input type " + fLayerDType
        )