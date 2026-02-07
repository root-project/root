from .. import get_keras_version


def MakeKerasFlatten(layer):
    """
    Create a Keras-compatible flattening operation using SOFIE framework.

    This function takes a dictionary representing a layer and its attributes and
    constructs a Keras-compatible flattening operation using the SOFIE framework.
    Flattening is the process of converting a multi-dimensional tensor into a
    one-dimensional tensor. Assumes layerDtype is float.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                name, data type, and other relevant information.

    Returns:
    ROperator_Reshape: A SOFIE framework operator representing the flattening operation.
    """
    from ROOT.TMVA.Experimental import SOFIE

    keras_version = get_keras_version()

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    attributes = layer["layerAttributes"]
    if keras_version < "2.16":
        flayername = attributes["_name"]
    else:
        flayername = attributes["name"]
    fOpMode = SOFIE.ReshapeOpMode.Flatten
    fNameData = finput[0]
    fNameOutput = foutput[0]
    fNameShape = flayername + "_shape"
    op = SOFIE.ROperator_Reshape(fOpMode, 0, fNameData, fNameShape, fNameOutput)
    return op
