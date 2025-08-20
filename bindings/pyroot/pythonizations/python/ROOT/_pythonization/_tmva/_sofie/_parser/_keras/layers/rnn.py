from cppyy import gbl as gbl_namespace

def MakeKerasRNN(layer): 
    """
    Create a Keras-compatible RNN (Recurrent Neural Network) layer operation using SOFIE framework.

    This function takes a dictionary representing an RNN layer and its attributes and
    constructs a Keras-compatible RNN layer operation using the SOFIE framework.
    RNN layers are used to model sequences, and they maintain internal states that are
    updated through recurrent connections.

    Parameters:
    layer (dict): A dictionary containing layer information including input, output,
                  layer type, attributes, weights, and data type - must be float.

    Returns:
    ROperator_RNN: A SOFIE framework operator representing the RNN layer operation.
    """
    
    # Extract required information from the layer dictionary
    fLayerDType = layer['layerDType']
    finput = layer['layerInput']
    foutput = layer['layerOutput']
    attributes = layer['layerAttributes']
    direction = attributes['direction']
    hidden_size = attributes["hidden_size"]
    layout = int(attributes["layout"])
    nameX = finput[0]
    nameY = foutput[0]
    nameW = layer["layerWeight"][0]
    nameR = layer["layerWeight"][1]
    if len(layer["layerWeight"]) > 2:
        nameB = layer["layerWeight"][2]
    else:
        nameB = ""
    
    # Check if the provided activation function is supported
    fPActivation = attributes['activation']
    if not fPActivation.__name__ in ['relu', 'sigmoid', 'tanh', 'softsign', 'softplus']: #avoiding functions with parameters
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator RNN does not yet support activation function " + fPActivation.__name__
        )
        
    activations = [fPActivation.__name__[0].upper()+fPActivation.__name__[1:]]

    #set default values
    activation_alpha = []
    activation_beta = []
    clip = 0.0
    nameY_h = ""
    nameInitial_h = ""
    name_seq_len = ""
    
    if  gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fLayerDType) ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
        if layer['layerType'] == "SimpleRNN":
            op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_RNN['float'](activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, nameX, nameW, nameR, nameB, name_seq_len, nameInitial_h, nameY, nameY_h)
        
        elif layer['layerType'] == "GRU":
            #an additional activation function is required, given by the user
            activations.insert(0, attributes['recurrent_activation'].__name__[0].upper() + attributes['recurrent_activation'].__name__[1:])
            
            #new variable needed:
            linear_before_reset = attributes['linear_before_reset']
            op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_GRU['float'](activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset, nameX, nameW, nameR, nameB, name_seq_len, nameInitial_h, nameY, nameY_h)
        
        elif layer['layerType'] == "LSTM":
            #an additional activation function is required, the first given by the user, the second set to tanh as default
            fPRecurrentActivation = attributes['recurrent_activation']
            if not fPActivation.__name__ in ['relu', 'sigmoid', 'tanh', 'softsign', 'softplus']: #avoiding functions with parameters
                raise RuntimeError(
                    "TMVA::SOFIE - Unsupported - Operator RNN does not yet support recurrent activation function " + fPActivation.__name__
                )
            fPRecurrentActivationName = fPRecurrentActivation.__name__[0].upper()+fPRecurrentActivation.__name__[1:]
            activations.insert(0,fPRecurrentActivationName)
            activations.insert(2,'Tanh')            
            
            #new variables needed:
            input_forget = 0
            nameInitial_c = ""
            nameP = "" #No peephole connections in keras LSTM model
            nameY_c = ""
            op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_LSTM['float'](activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout, nameX, nameW, nameR, nameB, name_seq_len, nameInitial_h, nameInitial_c, nameP, nameY, nameY_h, nameY_c)
        
        else: 
            raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator RNN does not yet support operator type " + layer['layerType']
        ) 
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator RNN does not yet support input type " + fLayerDType
        )   
