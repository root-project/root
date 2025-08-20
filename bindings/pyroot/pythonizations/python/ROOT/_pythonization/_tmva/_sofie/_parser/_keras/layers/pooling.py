from cppyy import gbl as gbl_namespace

def MakeKerasPooling(layer):
    """
    Create a Keras-compatible pooling layer operation using SOFIE framework.

    This function takes a dictionary representing a pooling layer and its attributes and
    constructs a Keras-compatible pooling layer operation using the SOFIE framework.
    Pooling layers downsample the input tensor by selecting a representative value from
    a group of neighboring values, either by taking the maximum or the average.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer type (the selection rule), the pool size, padding, strides, and data type.

    Returns:
    ROperator_Pool: A SOFIE framework operator representing the pooling layer operation.
    """
    
    #extract attributes from layer data
    fLayerDType = layer['layerDType']
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerType = layer['layerType']
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    pool_atrr = gbl_namespace.TMVA.Experimental.SOFIE.RAttributes_Pool()
    attributes = layer['layerAttributes']
    fAttrKernelShape = attributes["pool_size"]
    fKerasPadding = str(attributes["padding"])
    fAttrStrides = attributes["strides"]
    
    #Set default values
    fAttrDilations = (1,1)
    fpads = [0,0,0,0,0,0]
    pool_atrr.ceil_mode = 0
    pool_atrr.count_include_pad = 0
    pool_atrr.storage_order = 0
    
    if fKerasPadding == 'valid':
        fAttrAutopad = 'VALID'
    elif fKerasPadding == 'same':
        fAttrAutopad = 'NOTSET'
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet supports Convolution layer with padding " + fKerasPadding
        )
    pool_atrr.dilations = list(fAttrDilations)
    pool_atrr.strides = list(fAttrStrides)
    pool_atrr.pads = fpads
    pool_atrr.kernel_shape = list(fAttrKernelShape)
    pool_atrr.auto_pad = fAttrAutopad    
    
    #choose pooling type
    if 'Max' in fLayerType:
        PoolMode =  gbl_namespace.TMVA.Experimental.SOFIE.PoolOpMode.MaxPool
    elif 'AveragePool' in fLayerType:
        PoolMode =  gbl_namespace.TMVA.Experimental.SOFIE.PoolOpMode.AveragePool
    elif 'GlobalAverage' in fLayerType:
        PoolMode =  gbl_namespace.TMVA.Experimental.SOFIE.PoolOpMode.GloabalAveragePool
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator poolong does not yet support pooling type " + fLayerType
        )
    
    #create operator
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Pool['float'](PoolMode, pool_atrr, fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Pooling does not yet support input type " + fLayerDType
        )