#include "TMVA/RModelParser_Keras.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyKeras{

namespace INTERNAL{


void AddKerasLayer(RModel& rmodel, PyObject* fLayer){
   std::string fLayerType = PyStringAsString(PyDict_GetItemString(fLayer,"layerType"));

   //For layers without additional activation attribute
   auto findLayer = mapKerasLayer.find(fLayerType);
   if(findLayer != mapKerasLayer.end()){
      rmodel.AddOperator((findLayer->second)(fLayer));
      return;
   }

   //For layers like Dense & Conv which has additional activation attribute
   else if(mapKerasLayerWithActivation.find(fLayerType) != mapKerasLayerWithActivation.end()){
      findLayer = mapKerasLayerWithActivation.find(fLayerType);
      PyObject* fAttributes=PyDict_GetItemString(fLayer,"layerAttributes");

      std::string fLayerName = PyStringAsString(PyDict_GetItemString(fAttributes,"name"));
      std::string fLayerActivation = PyStringAsString(PyDict_GetItemString(fAttributes,"activation"));

      //Checking if additional attribute exixts
      if(fLayerActivation != "linear"){
         PyObject* fOutputs = PyDict_GetItemString(fLayer,"layerOutput");
         PyObject* fInputs = PyDict_GetItemString(fLayer,"layerInput");
         std::string fActivationLayerOutput = PyStringAsString(PyList_GetItem(fOutputs,0));

         PyList_SetItem(fOutputs,0,PyUnicode_FromString((fLayerName+fLayerType).c_str()));
         PyDict_SetItemString(fLayer,"layerOutput",fOutputs);
         rmodel.AddOperator((findLayer->second)(fLayer));

         std::string fActivationLayerInput = PyStringAsString(PyList_GetItem(fOutputs,0));
         PyList_SetItem(fInputs,0,PyUnicode_FromString(fActivationLayerInput.c_str()));
         PyList_SetItem(fOutputs,0,PyUnicode_FromString(fActivationLayerOutput.c_str()));
         PyDict_SetItemString(fLayer,"layerInput",fInputs);
         PyDict_SetItemString(fLayer,"layerOutput",fOutputs);

         auto findActivationLayer = mapKerasLayer.find(fLayerActivation);
         if(findActivationLayer == mapKerasLayer.end()){
            throw std::runtime_error("TMVA::SOFIE - Parsing Keras Activation layer " + fLayerActivation + " is not yet supported");
         }
         rmodel.AddOperator((findActivationLayer->second)(fLayer));
      }
      else{
         rmodel.AddOperator((findLayer->second)(fLayer));
      }
      return;
   }

   else{
      throw std::runtime_error("TMVA::SOFIE - Parsing Keras layer " + fLayerType + " is not yet supported");
   }

}


std::unique_ptr<ROperator> MakeKerasDense(PyObject* fLayer){
      PyObject* fInputs=PyDict_GetItemString(fLayer,"layerInput");
      PyObject* fOutputs=PyDict_GetItemString(fLayer,"layerOutput");
      std::string fLayerDType = PyStringAsString(PyDict_GetItemString(fLayer,"layerDType"));

      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      PyObject* fWeightNames = PyDict_GetItemString(fLayer,"layerWeight");
      std::string fKernelName = PyStringAsString(PyList_GetItem(fWeightNames,0));
      std::string fBiasName   = PyStringAsString(PyList_GetItem(fWeightNames,1));

      std::unique_ptr<ROperator> op;

      float attr_alpha = 1.0;
      float attr_beta  = 1.0;
      int_t attr_transA = 0;
      int_t attr_transB = 0;

      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, fLayerInputName, fKernelName, fBiasName, fLayerOutputName));
         break;

         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + fLayerDType);
         }
         return op;
}



std::unique_ptr<ROperator> MakeKerasActivation(PyObject* fLayer){
      PyObject* fAttributes=PyDict_GetItemString(fLayer,"layerAttributes");
      std::string fLayerActivation = PyStringAsString(PyDict_GetItemString(fAttributes,"activation"));

      auto findLayer = mapKerasLayer.find(fLayerActivation);
      if(findLayer == mapKerasLayer.end()){
         throw std::runtime_error("TMVA::SOFIE - Parsing Keras Activation layer " + fLayerActivation + " is not yet supported");
      }
      return (findLayer->second)(fLayer);
}



std::unique_ptr<ROperator> MakeKerasReLU(PyObject* fLayer)
{
      PyObject* fInputs=PyDict_GetItemString(fLayer,"layerInput");
      PyObject* fOutputs=PyDict_GetItemString(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(PyDict_GetItemString(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Relu<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + fLayerDType);
         }
   return op;
}


std::unique_ptr<ROperator> MakeKerasPermute(PyObject* fLayer)
{
      PyObject* fAttributes=PyDict_GetItemString(fLayer,"layerAttributes");
      PyObject* fInputs=PyDict_GetItemString(fLayer,"layerInput");
      PyObject* fOutputs=PyDict_GetItemString(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(PyDict_GetItemString(fLayer,"layerDType"));
      std::string fLayerInputName      = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName     = PyStringAsString(PyList_GetItem(fOutputs,0));

      PyObject* fAttributePermute=PyDict_GetItemString(fAttributes,"dims");
      std::vector<int_t>fPermuteDims;
      for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(fAttributePermute);++tupleIter){

         fPermuteDims.push_back((int_t)PyLong_AsLong(PyTuple_GetItem(fAttributePermute,tupleIter)));
      }
      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
            if (!fPermuteDims.empty()){
               op.reset(new ROperator_Transpose<float>(fPermuteDims, fLayerInputName, fLayerOutputName));
               }
            else{
               op.reset(new ROperator_Transpose<float> (fLayerInputName, fLayerOutputName));
               }
         break;
         default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + fLayerDType);
            }
   return op;
   }


std::vector<size_t> GetShapeFromTuple(PyObject* shapeTuple){
   std::vector<size_t>inputShape;
   for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(shapeTuple);++tupleIter){
               inputShape.push_back((size_t)PyLong_AsLong(PyTuple_GetItem(shapeTuple,tupleIter)));
         }
   return inputShape;
}
}//INTERNAL


RModel Parse(std::string filename){

   char sep = '/';
   #ifdef _WIN32
   sep = '\\';
   #endif

   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos){
      filename_nodir = (filename.substr(isep+1, filename.length() - isep));
   }

   //Check on whether the ONNX file exists
   if(!std::ifstream(filename).good()){
        throw std::runtime_error("Model file "+filename_nodir+" not found!");
    }


   std::time_t ttime = std::time(0);
   std::tm* gmt_time = std::gmtime(&ttime);
   std::string parsetime (std::asctime(gmt_time));

   RModel rmodel(filename_nodir, parsetime);

   //Intializing Python Interpreter and scope dictionaries
   Py_Initialize();
   PyObject* main = PyImport_AddModule("__main__");
   PyObject* fGlobalNS = PyModule_GetDict(main);
   PyObject* fLocalNS = PyDict_New();
   if (!fGlobalNS) {
       throw std::runtime_error("Can't init global namespace for Python");
       }
   if (!fLocalNS) {
       throw std::runtime_error("Can't init local namespace for Python");
       }

   //Extracting model information: For each layer: type,name,activation,dtype,input tensor's name,
   //output tensor's name, kernel's name, bias's name
   //None object is returned for if property doesn't belong to layer
   PyRunString("import tensorflow.keras as keras",fGlobalNS,fLocalNS);
   PyRunString("from tensorflow.keras.models import load_model",fGlobalNS,fLocalNS);
   PyRunString("print('Keras Version: '+ keras.__version__)",fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model=load_model('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model.load_weights('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
   PyRunString("modelData=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.layers)):\n"
            "  layer=model.get_layer(index=idx)\n"
            "  globals().update(locals())\n"
            "  layerData={}\n"
            "  layerData['layerType']=layer.__class__.__name__\n"
            "  layerData['layerAttributes']=layer.get_config()\n"
            "  layerData['layerInput']=[x.name for x in layer.input] if isinstance(layer.input,list) else [layer.input.name]\n"
            "  layerData['layerOutput']=[x.name for x in layer.output] if isinstance(layer.output,list) else [layer.output.name]\n"
            "  layerData['layerDType']=layer.dtype\n"
            "  layerData['layerWeight']=[x.name for x in layer.weights]\n"
            "  modelData.append(layerData)",fGlobalNS,fLocalNS);


   PyObject* fPModel = PyDict_GetItemString(fLocalNS,"modelData");
   PyObject *fLayer;
   Py_ssize_t fModelSize = PyList_Size(fPModel);
   std::string fLayerType;

   for(Py_ssize_t fModelIterator=0;fModelIterator<fModelSize;++fModelIterator){
      fLayer     = PyList_GetItem(fPModel,fModelIterator);
      fLayerType = PyStringAsString(PyDict_GetItemString(fLayer,"layerType"));

      //Ignoring the input layer for models built using Keras Functional API
      if(fLayerType == "InputLayer")
         continue;
      else if(fLayerType == "Dense")
         rmodel.AddBlasRoutines({"Gemm", "Gemv"});
      INTERNAL::AddKerasLayer(rmodel,fLayer);

   }

   //Extracting model's weights
   //For every initialized tensor, weightProp will have its name and dtype in string
   //and value in numpy array
   PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
   PyRunString("weight=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.get_weights())):\n"
               "  weightProp={}\n"
               "  weightProp['name']=model.weights[idx].name\n"
               "  weightProp['dtype']=(model.get_weights())[idx].dtype.name\n"
               "  weightProp['value']=(model.get_weights())[idx]\n"
               "  weight.append(weightProp)",fGlobalNS,fLocalNS);

   PyObject *fWeightTensor, *fPWeight;
   PyArrayObject *fWeightTensorValue;
   std::string fWeightName;
   ETensorType fWeightDType;
   fPWeight = PyDict_GetItemString(fLocalNS,"weight");
   std::vector<std::size_t> fWeightTensorShape;
   std::size_t fWeightTensorSize;

   for (Py_ssize_t weightIter = 0; weightIter < PyList_Size(fPWeight); weightIter++){
      fWeightTensor      = PyList_GetItem(fPWeight, weightIter);
      fWeightName        = PyStringAsString(PyDict_GetItemString(fWeightTensor,"name"));
      fWeightDType       = ConvertStringToType(PyStringAsString(PyDict_GetItemString(fWeightTensor,"dtype")));

      fWeightTensorValue = (PyArrayObject*)PyDict_GetItemString(fWeightTensor,"value");
      fWeightTensorSize=1;
      fWeightTensorShape.clear();
      for(int j=0; j<PyArray_NDIM(fWeightTensorValue); ++j){
       fWeightTensorShape.push_back((std::size_t)(PyArray_DIM(fWeightTensorValue,j)));
       fWeightTensorSize*=(std::size_t)(PyArray_DIM(fWeightTensorValue,j));
      }

   switch(fWeightDType){
       case ETensorType::FLOAT : {
       float* fWeightArray = (float*)PyArray_DATA(fWeightTensorValue);
       std::shared_ptr<void> fData(malloc(fWeightTensorSize * sizeof(float)), free);
       std::memcpy(fData.get(),fWeightArray, fWeightTensorSize * sizeof(float));
       rmodel.AddInitializedTensor(fWeightName,ETensorType::FLOAT,fWeightTensorShape,fData);
       break;
       }
       default:
          throw std::runtime_error("Type error: TMVA SOFIE does not yet weight data layer type"+ConvertTypeToString(fWeightDType));
      }
     }


   //Extracting input tensor info
   //For every input tensor inputNames will have their names as string,inputShapes will have their
   //shape as Python Tuple, and inputTypes will have their dtype as string
   PyRunString("inputNames=model.input_names",fGlobalNS,fLocalNS);
   PyRunString("inputShapes=model.input_shape",fGlobalNS,fLocalNS);
   PyRunString("inputTypes=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.inputs)):\n"
               "  inputTypes.append(model.inputs[idx].dtype.__str__()[9:-2])",fGlobalNS,fLocalNS);

   PyObject* fPInputs       = PyDict_GetItemString(fLocalNS,"inputNames");
   PyObject* fPInputShapes  = PyDict_GetItemString(fLocalNS,"inputShapes");
   PyObject* fPInputTypes   = PyDict_GetItemString(fLocalNS,"inputTypes");

   std::string fInputName;
   ETensorType fInputDType;

   //For single input models, the model.input_shape will return a tuple describing the input tensor shape
   //For multiple inputs models, the model.input_shape will return a list of tuple, each describing the input tensor shape.
   if(PyTuple_Check(fPInputShapes)){
      fInputName  = PyStringAsString(PyList_GetItem(fPInputs,0));
      fInputDType = ConvertStringToType(PyStringAsString(PyList_GetItem(fPInputTypes,0)));

      switch(fInputDType){

         case ETensorType::FLOAT : {
         if (PyTuple_GetItem(fPInputShapes,0) == Py_None){
            throw std::runtime_error("None error: Models not initialized with batch-size are not yet supported in TMVA SOFIE");
         }

         std::vector<size_t>fInputShape=INTERNAL::GetShapeFromTuple(fPInputShapes);
         rmodel.AddInputTensorInfo(fInputName, ETensorType::FLOAT, fInputShape);
         break;
         }

         default:
         throw std::runtime_error("Type error: TMVA SOFIE does not yet suppport data type"+ConvertTypeToString(fInputDType));
      }

   }

   else{

      for(Py_ssize_t inputIter = 0; inputIter < PyList_Size(fPInputs);++inputIter){

      fInputName  = PyStringAsString(PyList_GetItem(fPInputs,inputIter));
      fInputDType = ConvertStringToType(PyStringAsString(PyList_GetItem(fPInputTypes,inputIter)));

      switch(fInputDType){

         case ETensorType::FLOAT : {
         PyObject* fInputShapeTuple=PyList_GetItem(fPInputShapes,inputIter);

         if (PyTuple_GetItem(fInputShapeTuple,0) == Py_None){
            throw std::runtime_error("None error: Models not initialized with batch-size are not yet supported in TMVA SOFIE");
         }

         std::vector<size_t>fInputShape=INTERNAL::GetShapeFromTuple(fInputShapeTuple);
         rmodel.AddInputTensorInfo(fInputName, ETensorType::FLOAT, fInputShape);
         break;
         }

         default:
         throw std::runtime_error("Type error: TMVA SOFIE does not yet suppport data type"+ConvertTypeToString(fInputDType));

      }
      }
   }


   //Extracting Output Tensor Names
   PyRunString("outputNames=[]",fGlobalNS,fLocalNS);
   PyRunString("for layerName in model.output_names:\n"
               "    outputNames.append(model.get_layer(layerName).output.name)",fGlobalNS,fLocalNS);
   PyObject* fPOutputs   = PyDict_GetItemString(fLocalNS,"outputNames");
   std::vector<std::string> fOutputNames;
   for(Py_ssize_t outputIter = 0; outputIter < PyList_Size(fPOutputs);++outputIter){
         fOutputNames.push_back(PyStringAsString(PyList_GetItem(fPOutputs,outputIter)));
   }
   rmodel.AddOutputTensorNameList(fOutputNames);

   return rmodel;
}
}//PyKeras
}//SOFIE
}//Experimental
}//TMVA
