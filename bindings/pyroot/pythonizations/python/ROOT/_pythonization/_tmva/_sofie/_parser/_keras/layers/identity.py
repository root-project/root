def MakeKerasIdentity(layer):
    from ROOT.TMVA.Experimental import SOFIE

    input = layer["layerInput"]
    output = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = input[0]
    fLayerOutputName = output[0]
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_Identity("float")(fLayerInputName, fLayerOutputName)
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Identity does not yet support input type " + fLayerDType
        )
