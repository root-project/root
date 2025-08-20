from cppyy import gbl as gbl_namespace
from ..._keras import keras_version

def MakeKerasBatchNorm(layer): 
    """
    Create a Keras-compatible batch normalization operation using SOFIE framework.

    This function takes a dictionary representing a batch normalization layer and its
    attributes and constructs a Keras-compatible batch normalization operation using
    the SOFIE framework. Batch normalization is used to normalize the activations of
    a neural network, typically applied after the convolutional or dense layers.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  gamma, beta, moving mean, moving variance, epsilon,
                  momentum, data type (assumed to be float), and other relevant information.

    Returns:
    ROperator_BatchNormalization: A SOFIE framework operator representing the batch normalization operation.
    """
        
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    attributes = layer['layerAttributes']
    gamma = attributes["gamma"]
    beta = attributes["beta"]
    moving_mean = attributes["moving_mean"]
    moving_variance = attributes["moving_variance"]
    fLayerDType = layer["layerDType"]
    fNX = str(finput[0])
    fNY = str(foutput[0])
    
    if keras_version < '2.16':
        fNScale = gamma.name
        fNB = beta.name
        fNMean = moving_mean.name
        fNVar = moving_variance.name
    else:
        fNScale = gamma.path
        fNB = beta.path
        fNMean = moving_mean.path
        fNVar = moving_variance.path
        
    epsilon = attributes["epsilon"]
    momentum = attributes["momentum"]
    
    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_BatchNormalization('float')(epsilon, momentum, 0, fNX, fNScale, fNB, fNMean, fNVar, fNY)
    return op