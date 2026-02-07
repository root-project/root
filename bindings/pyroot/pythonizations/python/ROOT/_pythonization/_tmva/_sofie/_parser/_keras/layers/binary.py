def MakeKerasBinary(layer):
    from ROOT.TMVA.Experimental import SOFIE

    input = layer["layerInput"]
    output = layer["layerOutput"]
    fLayerType = layer["layerType"]
    fLayerDType = layer["layerDType"]
    fX1 = input[0]
    fX2 = input[1]
    fY = output[0]
    op = None
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        if fLayerType == "Add":
            op = SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Add)(fX1, fX2, fY)
        elif fLayerType == "Subtract":
            op = SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Sub)(fX1, fX2, fY)
        else:
            op = SOFIE.ROperator_BasicBinary(float, SOFIE.EBasicBinaryOperator.Mul)(fX1, fX2, fY)
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator BasicBinary does not yet support input type " + fLayerDType
        )
    return op
