from cppyy import gbl as gbl_namespace

def MakeKerasELU(layer):
    """
    Create a Keras-compatible exponential linear Unit (ELU) activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible ELU activation operation using the SOFIE framework.
    ELU is an activation function that modifies only the negative part of ReLU by 
    applying an exponential curve. It allows small negative values instead of zeros.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type, which must be float.

    Returns:
    ROperator_Elu: A SOFIE framework operator representing the ELU activation operation.
    """
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer['layerAttributes']
    if 'alpha' in attributes.keys():
        fAlpha = attributes['alpha']
    else:
        fAlpha = 1.0
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Elu('float')(fAlpha, fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + fLayerDType
        )
