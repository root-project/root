def MakeKerasBinary(layer):
    from ROOT.TMVA.Experimental import SOFIE

    inpt = layer["layerInput"]
    return SOFIE.createBasicBinary(layer["layerDType"], layer["layerType"], inpt[0], inpt[1], layer["layerOutput"][0])
