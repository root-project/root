from cppyy import gbl as gbl_namespace

def MakeKerasSoftmax(layer):
    """
    Create a Keras-compatible softmax activation operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible softmax activation operation using the SOFIE framework.
    Softmax is an activation function that converts input values into a probability
    distribution, often used in the output layer of a neural network for multi-class
    classification tasks.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  and data type - must be float.

    Returns:
    ROperator_Softmax: A SOFIE framework operator representing the softmax activation operation.
    """
    
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Softmax('float')(-1, fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Softmax does not yet support input type " + fLayerDType
        )