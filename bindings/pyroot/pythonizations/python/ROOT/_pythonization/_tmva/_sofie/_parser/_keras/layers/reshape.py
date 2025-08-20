from cppyy import gbl as gbl_namespace
from ..._keras import keras_version

def MakeKerasReshape(layer):
    """
    Create a Keras-compatible reshaping operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible reshaping operation using the SOFIE framework. Assumes layerDtype is float.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  name, data type, and other relevant information.

    Returns:
    ROperator_Reshape: A SOFIE framework operator representing the reshaping operation.
    """
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    attributes = layer['layerAttributes']
    if keras_version < '2.16':
        flayername = attributes['_name']
    else:
        flayername = attributes['name']
    fOpMode = gbl_namespace.TMVA.Experimental.SOFIE.ReshapeOpMode.Reshape
    fLayerDType = layer['layerDType']
    fNameData = finput[0]
    fNameOutput = foutput[0]
    fNameShape = flayername + "ReshapeAxes"
    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Reshape(fOpMode, 0, fNameData, fNameShape, fNameOutput)
    return op