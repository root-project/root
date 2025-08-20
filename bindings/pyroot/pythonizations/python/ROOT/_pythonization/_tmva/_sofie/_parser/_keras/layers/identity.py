from cppyy import gbl as gbl_namespace

def MakeKerasIdentity(layer):
    input = layer['layerInput']
    output = layer['layerOutput']
    fLayerDType = layer['layerDType']
    fLayerInputName = input[0]
    fLayerOutputName = output[0]
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Identity('float')(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Identity does not yet support input type " + fLayerDType
        )
