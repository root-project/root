def MakeKerasConcat(layer):
    from ROOT.TMVA.Experimental import SOFIE

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    attributes = layer["layerAttributes"]
    input = [str(i) for i in finput]
    output = str(foutput[0])
    axis = int(attributes["axis"])
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Concat(input, axis, 0, output)
    else:
        raise RuntimeError("TMVA::SOFIE - Unsupported - Operator Concat does not yet support input type " + fLayerDType)
    return op
