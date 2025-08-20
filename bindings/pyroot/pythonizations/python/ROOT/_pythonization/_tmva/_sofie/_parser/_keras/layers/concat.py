from cppyy import gbl as gbl_namespace

def MakeKerasConcat(layer):
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    attributes = layer['layerAttributes']
    input = [str(i) for i in finput]
    output = str(foutput[0])
    axis = int(attributes["axis"])
    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Concat(input, axis, 0,  output)
    return op