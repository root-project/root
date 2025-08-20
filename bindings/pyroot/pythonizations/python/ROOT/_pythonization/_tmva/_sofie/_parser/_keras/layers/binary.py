from cppyy import gbl as gbl_namespace

def MakeKerasBinary(layer):
    input = layer['layerInput']
    output = layer['layerOutput']
    fLayerType = layer['layerType'] 
    fLayerDType = layer['layerDType'] 
    fX1 = input[0]
    fX2 = input[1]
    fY = output[0]
    op = None
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        if fLayerType == "Add":
          op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_BasicBinary(float,'TMVA::Experimental::SOFIE::EBasicBinaryOperator::Add')(fX1, fX2, fY)
        elif fLayerType == "Subtract":
          op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_BasicBinary(float,'TMVA::Experimental::SOFIE::EBasicBinaryOperator::Sub')(fX1, fX2, fY)
        else:
          op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_BasicBinary(float,'TMVA::Experimental::SOFIE::EBasicBinaryOperator::Mul')(fX1, fX2, fY)
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator Identity does not yet support input type " + fLayerDType
        )
    return op