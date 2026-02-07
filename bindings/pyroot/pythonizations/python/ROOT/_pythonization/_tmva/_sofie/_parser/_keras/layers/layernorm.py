from .. import get_keras_version


def MakeKerasLayerNorm(layer):
    """
    Create a Keras-compatible layer normalization operation using SOFIE framework.

    This function takes a dictionary representing a layer normalization layer and its
    attributes and constructs a Keras-compatible layer normalization operation using
    the SOFIE framework. Unlike Batch normalization, Layer normalization used to normalize 
    the activations of a layer across the entire layer, independently for each sample in 
    the batch.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  gamma, beta, epsilon, data type (assumed to be float), and other 
                  relevant information.

    Returns:
    ROperator_BatchNormalization: A SOFIE framework operator representing the layer normalization operation.
    """
    from ROOT.TMVA.Experimental import SOFIE
    
    keras_version = get_keras_version()
    
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    attributes = layer['layerAttributes']
    gamma = attributes["gamma"]
    beta = attributes["beta"]
    axes = attributes['axis']
    if '_build_input_shape' in attributes.keys():
        num_input_shapes = len(attributes['_build_input_shape'])
    elif '_build_shapes_dict' in attributes.keys():
        num_input_shapes = len(list(attributes['_build_shapes_dict']['input_shape']))
    if len(axes) == 1:
        axis = axes[0]
        if axis < 0:
            axis += num_input_shapes
    else:
        raise Exception("TMVA.SOFIE - LayerNormalization layer - parsing different axes at once is not supported")
    fLayerDType = layer["layerDType"]
    fNX = str(finput[0])
    fNY = str(foutput[0])        
    
    if keras_version < '2.16':
        fNScale = gamma.name
        fNB = beta.name
    else:
        fNScale = gamma.path
        fNB = beta.path
        
    epsilon = attributes["epsilon"]
    fNInvStdDev = []
    
    if  SOFIE.ConvertStringToType(fLayerDType) ==  SOFIE.ETensorType.FLOAT:
        op =  SOFIE.ROperator_LayerNormalization('float')(axis, epsilon, 1, fNX, fNScale, fNB, fNY, "", fNInvStdDev)
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator BatchNormalization does not yet support input type " + fLayerDType
        )
    
    return op
