from cppyy import gbl as gbl_namespace

def MakeKerasSeLU(layer):
    """
    Create a Keras-compatible scaled exponential linear unit (SeLU) activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible SeLU activation operation using the SOFIE framework.
    SeLU is a type of activation function that introduces self-normalizing properties
    to the neural network.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float32.

    Returns:
    ROperator_Selu: A SOFIE framework operator representing the SeLU activation operation.
    """
        
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Selu('float')(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Selu does not yet support input type " + fLayerDType
        )