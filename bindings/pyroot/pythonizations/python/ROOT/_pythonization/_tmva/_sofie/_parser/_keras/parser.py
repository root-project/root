from ......_pythonization import pythonization
from cppyy import gbl as gbl_namespace
import keras
import numpy as np
import os
import time

from .layers.permute import MakeKerasPermute
from .layers.batchnorm import MakeKerasBatchNorm
from .layers.reshape import MakeKerasReshape
from .layers.flatten import MakeKerasFlatten
from .layers.concat import MakeKerasConcat
from .layers.swish import MakeKerasSwish
from .layers.binary import MakeKerasBinary
from .layers.softmax import MakeKerasSoftmax
from .layers.tanh import MakeKerasTanh
from .layers.identity import MakeKerasIdentity
from .layers.relu import MakeKerasReLU
from .layers.selu import MakeKerasSeLU
from .layers.sigmoid import MakeKerasSigmoid
from .layers.leakyrelu import MakeKerasLeakyRelu
from .layers.pooling import MakeKerasPooling
from .layers.rnn import MakeKerasRNN
from .layers.dense import MakeKerasDense
from .layers.conv import MakeKerasConv

from . import keras_version

def MakeKerasActivation(layer):
    attributes = layer['layerAttributes']
    activation = attributes['activation']
    fLayerActivation = str(activation.__name__)
        
    if fLayerActivation in mapKerasLayer.keys():
        return mapKerasLayer[fLayerActivation](layer)
    else:
        raise Exception("TMVA.SOFIE - parsing keras activation layer " + fLayerActivation + " is not yet supported")

# Set global dictionaries, mapping layers to corresponding functions that create their ROperator instances
mapKerasLayer = {"Activation": MakeKerasActivation,
                 "Permute": MakeKerasPermute,
                 "BatchNormalization": MakeKerasBatchNorm,
                 "Reshape": MakeKerasReshape,
                 "Flatten": MakeKerasFlatten,
                 "Concatenate": MakeKerasConcat,
                 "swish": MakeKerasSwish,
                 "silu": MakeKerasSwish,
                 "Add": MakeKerasBinary,
                 "Subtract": MakeKerasBinary,
                 "Multiply": MakeKerasBinary,
                 "Softmax": MakeKerasSoftmax,
                 "tanh": MakeKerasTanh,
                 "Identity": MakeKerasIdentity,
                 "Dropout": MakeKerasIdentity,
                 "ReLU": MakeKerasReLU,
                 "relu": MakeKerasReLU,
                 "selu": MakeKerasSeLU,
                 "sigmoid": MakeKerasSigmoid,
                 "LeakyReLU": MakeKerasLeakyRelu, 
                 "softmax": MakeKerasSoftmax, 
                 "MaxPooling2D": MakeKerasPooling,
                 "SimpleRNN": MakeKerasRNN,
                 "GRU": MakeKerasRNN,
                 "LSTM": MakeKerasRNN,
                 }

mapKerasLayerWithActivation = {"Dense": MakeKerasDense,"Conv2D": MakeKerasConv}

def add_layer_into_RModel(rmodel, layer_data):
    """
    Add a Keras layer operation to an existing RModel using the SOFIE framework.

    This function takes an existing RModel and a dictionary representing a Keras layer
    and its attributes, and adds the corresponding layer operation to the RModel using
    the SOFIE framework. The function supports various types of Keras layers, including
    those with or without activation functions.

    Parameters:
    rmodel (RModel): An existing RModel to which the layer operation will be added.
    layer_data (dict): A dictionary containing layer information including type,
                      attributes, input, output, and layer data type.

    Returns:
    RModel: The updated RModel after adding the layer operation.

    Raises exception: If the provided layer type or activation function is not supported.
    """
    
    fLayerType = layer_data['layerType']
    
    # reshape and flatten layers don't have weights, but they are needed inside the list of initialized 
    # tensor list in the Rmodel
    if fLayerType == "Reshape" or fLayerType == "Flatten":
        Attributes = layer_data['layerAttributes']
        if keras_version < '2.16':
            LayerName = Attributes['_name']
        else:
            LayerName = Attributes['name']
            
        if fLayerType == "Reshape":
            TargetShape = np.asarray(Attributes['target_shape']).astype("int")
            TargetShape = np.insert(TargetShape,0,0)
        else:
            if '_build_input_shape' in Attributes.keys():
                input_shape = Attributes['_build_input_shape']
            elif '_build_shapes_dict' in Attributes.keys():
                input_shape = list(Attributes['_build_shapes_dict']['input_shape'])
            else:
                raise RuntimeError (
                    "Failed to extract build input shape from " + fLayerType + " layer"
                )
            TargetShape = [ gbl_namespace.TMVA.Experimental.SOFIE.ConvertShapeToLength(input_shape[1:])]
            TargetShape = np.asarray(TargetShape)
        
        # since the AddInitializedTensor method in RModel requires unique pointer, we call a helper function 
        # in c++ that does the conversion from a regular pointer to unique one in c++
        rmodel.AddInitializedTensor['long'](LayerName+"ReshapeAxes", [len(TargetShape)], TargetShape)
    
    # These layers only have one operator - excluding the recurrent layers, in which the activation function(s) 
    # are included in the recurrent operator
    if fLayerType in mapKerasLayer.keys():
        Attributes = layer_data['layerAttributes']
        inputs = layer_data['layerInput']
        outputs = layer_data['layerOutput']
        if keras_version < '2.16':
            LayerName = Attributes['_name']
        else:
            LayerName = Attributes['name']
        
        # Pooling layers in keras by default assume the channels dimension is the last one, 
        # while in onnx (and the SOFIE's RModel) it is the first one (other than batch size), 
        # so a transpose is needed before and after the pooling, if the data format is channels 
        # last (can be set to channels first by the user). In case of MaxPool2D and Conv2D (with
        # linear activation) channels last, the transpose layers are added as:
        #                   input                   output
        # transpose layer   input_layer_name        layer_name + PreTrans
        # actual layer      layer_name + PreTrans   layer_name + PostTrans
        # transpose layer   layer_name + PostTrans  output_layer_name
        
        fLayerOutput = outputs[0]
        if fLayerType == 'MaxPooling2D':
            if layer_data['channels_last']:
                op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,3,1,2], inputs[0],
                                                                                         LayerName+"PreTrans")
                rmodel.AddOperatorReference(op)
                inputs[0] = LayerName+"PreTrans"
                layer_data["layerInput"] = inputs
                outputs[0] = LayerName+"PostTrans"
        rmodel.AddOperatorReference(mapKerasLayer[fLayerType](layer_data))
        if fLayerType == 'MaxPooling2D':
            if layer_data['channels_last']:
                op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,2,3,1], 
                                                                                    LayerName+"PostTrans", fLayerOutput)
                rmodel.AddOperatorReference(op)
        return rmodel
    
    # These layers require two operators - dense/conv and their activation function
    elif fLayerType in mapKerasLayerWithActivation.keys():
        Attributes = layer_data['layerAttributes']
        if keras_version < '2.16':
            LayerName = Attributes['_name']
        else:
            LayerName = Attributes['name']
        fPActivation = Attributes['activation']
        LayerActivation = fPActivation.__name__
        if LayerActivation in ['selu', 'sigmoid']:
            rmodel.AddNeededStdLib("cmath")
        
        # if there is an activation function after the layer
        if LayerActivation != 'linear':
            if not LayerActivation in mapKerasLayer.keys():
                raise Exception("TMVA.SOFIE - parsing keras activation function " + LayerActivation + " is not yet supported")
            outputs = layer_data['layerOutput']
            inputs = layer_data['layerInput']
            fActivationLayerOutput = outputs[0]
            
            # like pooling, convolutional layer from keras requires transpose before and after to match
            # the onnx format 
            # if the data format is channels last (can be set to channels first by the user).
            if fLayerType == 'Conv2D':
                if layer_data['channels_last']:
                    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,3,1,2], inputs[0], LayerName+"PreTrans")
                    rmodel.AddOperatorReference(op)
                    inputs[0] = LayerName+"PreTrans"
                    layer_data["layerInput"] = inputs
            outputs[0] = LayerName+fLayerType
            layer_data['layerOutput'] = outputs
            op = mapKerasLayerWithActivation[fLayerType](layer_data)
            rmodel.AddOperatorReference(op)
            Activation_layer_input = LayerName+fLayerType
            if fLayerType == 'Conv2D':
                if layer_data['channels_last']:
                    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,2,3,1], LayerName+fLayerType, LayerName+"PostTrans")
                    rmodel.AddOperatorReference(op)
                    Activation_layer_input = LayerName + "PostTrans"
            
            # Adding the activation function
            inputs[0] = Activation_layer_input
            outputs[0] = fActivationLayerOutput
            layer_data['layerInput'] = inputs
            layer_data['layerOutput'] = outputs
            
            rmodel.AddOperatorReference(mapKerasLayer[LayerActivation](layer_data))
            
        else: # if layer is conv and the activation is linear, we need to add transpose before and after
            if fLayerType == 'Conv2D':
                inputs = layer_data['layerInput']
                outputs = layer_data['layerOutput']
                fLayerOutput = outputs[0]
                if layer_data['channels_last']:
                    op = gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,3,1,2], inputs[0], LayerName+"PreTrans")
                    rmodel.AddOperatorReference(op)
                    inputs[0] = LayerName+"PreTrans"
                    layer_data['layerInput'] = inputs
                    outputs[0] = LayerName+"PostTrans"
            rmodel.AddOperatorReference(mapKerasLayerWithActivation[fLayerType](layer_data))
            if fLayerType == 'Conv2D':
                if layer_data['channels_last']:
                    op =  gbl_namespace.TMVA.Experimental.SOFIE.ROperator_Transpose('float')([0,2,3,1], LayerName+"PostTrans", fLayerOutput)
                    rmodel.AddOperatorReference(op)
        return rmodel
    else:
        raise Exception("TMVA.SOFIE - parsing keras layer " + fLayerType + " is not yet supported")

class RModelParser_Keras:

    def Parse(filename, batch_size=1):  # If a model does not have a defined batch size, then assuming it is 1
        #Check if file exists
        if not os.path.exists(filename):
            raise RuntimeError("Model file {} not found!".format(filename))
            
        # load model
        keras_model = keras.models.load_model(filename)
        keras_model.load_weights(filename)
        
        # create new RModel object
        sep = '/'
        if os.name == 'nt':
            sep = '\\'
        
        isep = filename.rfind(sep)
        filename_nodir = filename
        if isep != -1:
            filename_nodir = filename[isep+1:]
        
        ttime = time.time()
        gmt_time = time.gmtime(ttime)
        parsetime = time.asctime(gmt_time)
        
        rmodel = gbl_namespace.TMVA.Experimental.SOFIE.RModel.RModel(filename_nodir, parsetime)
        
        # iterate over the layers and add them to the RModel
        # in case of keras 3.x (particularly in sequential models), the layer input and output name conventions are 
        # different from keras 2.x. In keras 2.x, the layer input name is consistent with previous layer's output 
        # name. For e.g., if the sequence of layers is dense -> maxpool, the input and output layer names would be:
        #           layer   |       name
        # input     dense   |   keras_tensor_1
        # output    dense   |   keras_tensor_2 --
        #                   |                    |=> layer name matches
        # input     maxpool |   keras_tensor_2 --
        # output    maxpool |   keras_tensor_3
        #
        # but in case of keras 3.x, this changes.
        #           layer   |       name
        # input     dense   |   keras_tensor_1
        # output    dense   |   keras_tensor_2 --
        #                   |                    |=> different layer name 
        # input     maxpool |   keras_tensor_3 --
        # output    maxpool |   keras_tensor_4  
        #
        # hence, we need to add a custom layer iterator, which would replace the suffix of the layer's input
        # and output names
        layer_iter = 0     
        is_functional_model = True if keras_model.__class__.__name__ == 'Functional' else False
        
        for layer in keras_model.layers:
            layer_data={}
            layer_data['layerType']=layer.__class__.__name__
            layer_data['layerAttributes']=layer.__dict__
            if keras_version < '2.16' or is_functional_model:
                if 'input_layer' in layer.name:
                    layer_data['layerInput'] = layer.name
                else:
                    layer_data['layerInput']=[x.name for x in layer.input] if isinstance(layer.input,list) else [layer.input.name]
            else:
                if 'input_layer' in layer.input.name:
                    layer_data['layerInput'] = [layer.input.name]
                else:
                    input_layer_name = layer.input.name[:13] + str(layer_iter)
                    layer_data['layerInput'] = [input_layer_name]
            if keras_version < '2.16' or is_functional_model:
                layer_data['layerOutput']=[x.name for x in layer.output] if isinstance(layer.output,list) else [layer.output.name]
            else:
                output_layer_name = layer.output.name[:13] + str(layer_iter+1)
                layer_data['layerOutput']=[x.name for x in layer.output] if isinstance(layer.output,list) else [output_layer_name]
                layer_iter += 1
            
            layer_data['layerDType']=layer.dtype
            
            if len(layer.weights) > 0:
                if keras_version < '2.16':
                    layer_data['layerWeight'] = [x.name for x in layer.weights]
                else:
                    layer_data['layerWeight'] = [x.path for x in layer.weights]
            else:
                layer_data['layerWeight'] = []
            
            # for convolutional and pooling layers we need to know the format of the data
            if layer_data['layerType'] in ['Conv2D', 'MaxPooling2D']:
                layer_data['channels_last'] = True if layer.data_format == 'channels_last' else False
                
            # for recurrent type layers we need to extract additional unique information
            if layer_data['layerType'] in ["SimpleRNN", "LSTM", "GRU"]:
                layer_data['layerAttributes']['activation'] = layer.activation
                layer_data['layerAttributes']['direction'] = 'backward' if layer.go_backwards else 'forward'
                layer_data['layerAttributes']["units"] = layer.units
                layer_data['layerAttributes']["layout"] = layer.input.shape[0] is None
                layer_data['layerAttributes']["hidden_size"] = layer.output.shape[-1]
                
                # for GRU and LSTM we need to extract an additional activation function
                if layer_data['layerType'] != "SimpleRNN": 
                    layer_data['layerAttributes']['recurrent_activation'] = layer.recurrent_activation
                
                # for GRU there are two variants of the reset gate location, we need to know which one is it
                if layer_data['layerType'] == "GRU":
                    layer_data['layerAttributes']['linear_before_reset'] = 1 if layer.reset_after and layer.recurrent_activation.__name__ == "sigmoid" else 0
            
            fLayerType = layer_data['layerType']
            # Ignoring the input layer of the model
            if(fLayerType == "InputLayer"):
                continue;

            # Adding any required routines depending on the Layer types for generating inference code.
            if (fLayerType == "Dense"):
                rmodel.AddBlasRoutines({"Gemm", "Gemv"})
            elif (fLayerType == "BatchNormalization"):
                rmodel.AddBlasRoutines({"Copy", "Axpy"})
            elif (fLayerType == "Conv1D" or fLayerType == "Conv2D" or fLayerType == "Conv3D"):
                rmodel.AddBlasRoutines({"Gemm", "Axpy"})
            rmodel = add_layer_into_RModel(rmodel, layer_data)

        # Extracting model's weights
        weight = []
        for idx in range(len(keras_model.get_weights())):
            weightProp = {}
            if keras_version < '2.16':
                weightProp['name'] = keras_model.weights[idx].name
            else:
                weightProp['name'] = keras_model.weights[idx].path
            weightProp['dtype'] = keras_model.get_weights()[idx].dtype.name
            if 'conv' in weightProp['name'] and keras_model.weights[idx].shape.ndims == 4:
                weightProp['value'] = keras_model.get_weights()[idx].transpose((3, 2, 0, 1)).copy()
            else:
                weightProp['value'] = keras_model.get_weights()[idx]
            weight.append(weightProp)

        # Traversing through all the Weight tensors
        for weightIter in range(len(weight)):
            fWeightTensor = weight[weightIter]
            fWeightName = fWeightTensor['name']
            fWeightDType = gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fWeightTensor['dtype'])
            fWeightTensorValue = fWeightTensor['value']
            fWeightTensorSize = 1
            fWeightTensorShape = []
            
            #IS IT BATCH SIZE? CHECK ONNX
            if 'simple_rnn' in fWeightName or 'lstm' in fWeightName or ('gru' in fWeightName and not 'bias' in fWeightName):
                fWeightTensorShape.append(1)
            
            # Building the shape vector and finding the tensor size
            for j in range(len(fWeightTensorValue.shape)):
                fWeightTensorShape.append(fWeightTensorValue.shape[j])
                fWeightTensorSize *= fWeightTensorValue.shape[j]
            
            if fWeightDType ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
                fWeightArray = fWeightTensorValue
                
                # weights conversion format between keras and onnx for lstm: the order of the different 
                # elements (input, output, forget, cell) inside the vector/matrix is different
                if 'lstm' in fWeightName:
                    if 'kernel' in fWeightName:
                        units = int(fWeightArray.shape[1]/4)
                        W_i = fWeightArray[:, :units].copy()
                        W_f = fWeightArray[:, units: units * 2].copy()
                        W_c = fWeightArray[:, units * 2: units * 3].copy()
                        W_o = fWeightArray[:, units * 3:].copy()
                        fWeightArray[:, units: units * 2] = W_o
                        fWeightArray[:, units * 2: units * 3] = W_f
                        fWeightArray[:, units * 3:] = W_c
                    else: #bias
                        units = int(fWeightArray.shape[0]/4)
                        W_i = fWeightArray[:units].copy()
                        W_f = fWeightArray[units: units * 2].copy()
                        W_c = fWeightArray[units * 2: units * 3].copy()
                        W_o = fWeightArray[units * 3:].copy()
                        fWeightArray[units: units * 2] = W_o
                        fWeightArray[units * 2: units * 3] = W_f
                        fWeightArray[units * 3:] = W_c
            
                # need to make specific adjustments for recurrent weights and biases
                if ('simple_rnn' in fWeightName or 'lstm' in fWeightName or 'gru' in fWeightName):
                    # reshaping weight matrices for recurrent layers due to keras-onnx inconsistencies
                    if 'kernel' in fWeightName:
                        fWeightArray = np.transpose(fWeightArray)
                        fWeightTensorShape[1], fWeightTensorShape[2] = fWeightTensorShape[2], fWeightTensorShape[1]
                    
                    fData = fWeightArray.flatten()
                    
                    # the recurrent bias and the cell bias can be the same, in which case we need to add a 
                    # vector of zeros for the recurrent bias
                    if 'bias' in fWeightName and len(fData.shape) == 1:
                        fWeightTensorShape[1] *= 2
                        fRbias = fData.copy()*0
                        fData = np.concatenate((fData,fRbias))

                else:
                    fData = fWeightArray.flatten()
                rmodel.AddInitializedTensor['float'](fWeightName, fWeightTensorShape, fData)
            else:
                raise TypeError("Type error: TMVA SOFIE does not yet support data layer type: " + fWeightDType)
        
        # Extracting input tensor info
        if keras_version < '2.16':
            fPInputs = keras_model.input_names
        else:
            fPInputs = [x.name for x in keras_model.inputs]
            
        fPInputShape = keras_model.input_shape if isinstance(keras_model.input_shape, list) else [keras_model.input_shape]
        fPInputDType = []
        for idx in range(len(keras_model.inputs)):
            dtype = keras_model.inputs[idx].dtype.__str__()
            if (dtype == "float32"):
                fPInputDType.append(dtype)
            else:
                fPInputDType.append(dtype[9:-2])
        
        if len(fPInputShape) == 1:
            fInputName = fPInputs[0]
            fInputDType = gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fPInputDType[0])
            if fInputDType ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
                if fPInputShape[0][0] is None or fPInputShape[0][0] <= 0:
                    fPInputShape = list(fPInputShape[0])
                    fPInputShape[0] = batch_size
                rmodel.AddInputTensorInfo(fInputName, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, fPInputShape)
                rmodel.AddInputTensorName(fInputName) 
            else:
                raise TypeError("Type error: TMVA SOFIE does not yet support data type " + gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fInputDType))
        else:
            # Iterating through multiple input tensors
            for fInputName, fInputDType, fInputShapeTuple in zip(fPInputs, fPInputDType, fPInputShape):
                fInputDType = gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fInputDType)
                if fInputDType ==  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT:
                    if fInputShapeTuple[0] is None or fInputShapeTuple[0] <= 0:
                        fInputShapeTuple = list(fInputShapeTuple)
                        fInputShapeTuple[0] = batch_size
                    rmodel.AddInputTensorInfo(fInputName,  gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, fInputShapeTuple)
                    rmodel.AddInputTensorName(fInputName)
                else:
                    raise TypeError("Type error: TMVA SOFIE does not yet support data type " + gbl_namespace.TMVA.Experimental.SOFIE.ConvertStringToType(fInputDType))             
        
        # Adding OutputTensorInfos
        outputNames = []
        if keras_version < '2.16' or is_functional_model:
            for layerName in keras_model.output_names:
                output_layer= keras_model.get_layer(layerName)
                output_layer_name = output_layer.output.name
                outputNames.append(output_layer_name)
        else:
            output_layer = keras_model.layers[-1]
            output_layer.name = output_layer.name[:13] + str(layer_iter)
            outputNames.append(output_layer_name)
        rmodel.AddOutputTensorNameList(outputNames)
        return rmodel

@pythonization("RModelParser_Keras", ns="TMVA::Experimental::SOFIE")
def pythonize_rmodelparser_keras(klass):
    # Parameters:
    # klass: class to be pythonized 
    setattr(klass, "Parse", RModelParser_Keras.Parse)