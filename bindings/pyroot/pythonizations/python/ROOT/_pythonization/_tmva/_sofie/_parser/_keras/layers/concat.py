from cppyy import gbl as gbl_namespace

def MakeKerasConcat(layer):
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    fLayerDType = layer["layerDType"]
    attributes = layer['layerAttributes']
    input = [str(i) for i in finput]
    output = str(foutput[0])
    axis = int(attributes["axis"])
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Concat(input, axis, 0,  output)
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Concat does not yet support input type " + fLayerDType
        )
    return op