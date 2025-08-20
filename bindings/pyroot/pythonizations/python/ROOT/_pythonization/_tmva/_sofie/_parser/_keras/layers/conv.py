from cppyy import gbl as gbl_namespace
import math
from ..._keras import keras_version

def MakeKerasConv(layer): 
    """
    Create a Keras-compatible convolutional layer operation using SOFIE framework.

    This function takes a dictionary representing a convolutional layer and its attributes and
    constructs a Keras-compatible convolutional layer operation using the SOFIE framework.
    A convolutional layer applies a convolution operation between the input tensor and a set
    of learnable filters (kernels).

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  data type (must be float), weight and bias name, kernel size, dilations, padding and strides. 
                  When padding is same (keep in the same dimensions), the padding shape is calculated.

    Returns:
    ROperator_Conv: A SOFIE framework operator representing the convolutional layer operation.
    """
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer['layerAttributes']
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1]
    fAttrDilations = attributes["dilation_rate"]
    fAttrGroup = int(attributes["groups"])
    fAttrKernelShape = attributes["kernel_size"]
    fKerasPadding = str(attributes["padding"])
    fAttrStrides = attributes["strides"]
    fAttrPads = []
    
    if fKerasPadding == 'valid':
        fAttrAutopad = 'VALID'
    elif fKerasPadding == 'same':
        fAttrAutopad = 'NOTSET'
        if keras_version < '2.16':
            fInputShape = attributes['_build_input_shape']
        else:
            fInputShape = attributes['_build_shapes_dict']['input_shape']
        inputHeight = fInputShape[1]
        inputWidth = fInputShape[2]
        outputHeight = math.ceil(float(inputHeight) / float(fAttrStrides[0]))
        outputWidth = math.ceil(float(inputWidth) / float(fAttrStrides[1]))
        padding_height = max((outputHeight - 1) * fAttrStrides[0] + fAttrKernelShape[0] - inputHeight, 0)
        padding_width = max((outputWidth - 1) * fAttrStrides[1] + fAttrKernelShape[1] - inputWidth, 0)
        padding_top = math.floor(padding_height / 2)
        padding_bottom = padding_height - padding_top
        padding_left = math.floor(padding_width / 2)
        padding_right = padding_width - padding_left
        fAttrPads = [padding_top, padding_bottom, padding_left, padding_right]
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet supports Convolution layer with padding " + fKerasPadding
        )
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Conv['float'](fAttrAutopad, fAttrDilations, fAttrGroup, 
                                                                  fAttrKernelShape, fAttrPads, fAttrStrides, 
                                                                  fLayerInputName, fKernelName, fBiasName, 
                                                                  fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + fLayerDType
        )