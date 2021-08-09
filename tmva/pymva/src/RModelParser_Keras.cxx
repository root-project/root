#include "TMVA/RModelParser_Keras.h"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


namespace INTERNAL{

   std::unique_ptr<ROperator> make_ROperator_Gemm(std::string input,std::string output,std::string kernel,std::string bias, ETensorType dtype)
   {
      std::unique_ptr<ROperator> op;

      float attr_alpha = 1.0;
      float attr_beta  = 1.0;
      int_t attr_transA = 0;
      int_t attr_transB = 0;

      switch(dtype){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, input, kernel, bias, output));
         break;

         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + ConvertTypeToString(dtype));
         }

         return op;
   }

   std::unique_ptr<ROperator> make_ROperator_Relu(std::string input, std::string output, ETensorType dtype)
   {
      std::unique_ptr<ROperator> op;
      switch(dtype){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Relu<float>(input, output));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + ConvertTypeToString(dtype));
         }
   return op;
   }

   std::unique_ptr<ROperator> make_ROperator_Transpose(std::string input, std::string output, std::vector<int_t> dims, ETensorType dtype)
   {
      std::unique_ptr<ROperator> op;
      std::vector<int_t> attr_perm=dims;
      switch(dtype){
         case ETensorType::FLOAT:
            if (!attr_perm.empty()){
               op.reset(new ROperator_Transpose<float>(attr_perm, input, output));
               }
            else{
               op.reset(new ROperator_Transpose<float> (input, output));
               }
         break;
         default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + ConvertTypeToString(dtype));
            }
   return op;
   }

}

namespace PyKeras{

void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS){
   PyObject *fPyReturn = PyRun_String(code, Py_single_input, fGlobalNS, fLocalNS);
   if (!fPyReturn) {
      std::cout<<"Python error message:\n";
      PyErr_Print();
      throw std::runtime_error("Failed to run python code: "+code);
   }
 }

const char* PyStringAsString(PyObject* str){
   #if PY_MAJOR_VERSION < 3   // for Python2
      const char *stra_name = PyBytes_AsString(str);
      // need to add string delimiter for Python2
      TString sname = TString::Format("'%s'",stra_name);
      const char * name = sname.Data();
   #else   // for Python3
      PyObject* repr = PyObject_Repr(str);
      PyObject* stra = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
      const char *name = PyBytes_AsString(stra);
   #endif
return name;
}

std::vector<size_t> GetShapeFromTuple(PyObject* shapeTuple){
   std::vector<size_t>inputShape;
   for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(shapeTuple);++tupleIter){
               inputShape.push_back((size_t)PyLong_AsLong(PyTuple_GetItem(shapeTuple,tupleIter)));
         }
   return inputShape;
}


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
   PyRunString("import keras",fGlobalNS,fLocalNS);
   PyRunString("from keras.models import load_model",fGlobalNS,fLocalNS);
   PyRunString("print('Keras Version: '+ keras.__version__)",fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model=load_model('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model.load_weights('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
   PyRunString("modelData=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.layers)):\n"
            "  layer=model.get_layer(index=idx)\n"
            "  globals().update(locals())\n"
            "  layerData=[]\n"
            "  layerData.append(layer.__class__.__name__)\n"
            "  layerData.append(layer.get_config())\n"
            "  layerData.append(layer.input)\n"
            "  layerData.append(layer.output)\n"
            "  layerData.append([x.name for x in layer.weights])\n"
            "  modelData.append(layerData)",fGlobalNS,fLocalNS);


   PyObject* fPModel = PyDict_GetItemString(fLocalNS,"modelData");
   PyObject *fLayer,*fAttributes,*fInputs,*fOutputs, *fWeightsNames;
   Py_ssize_t fModelSize = PyList_Size(fPModel);
   std::string fLayerName, fLayerType, fLayerActivation, fLayerInput, fLayerOutput;
   ETensorType fLayerDType;

   for(Py_ssize_t fModelIterator=0;fModelIterator<fModelSize;++fModelIterator){
      fLayer     = PyList_GetItem(fPModel,fModelIterator);
      fLayerType = PyStringAsString(PyList_GetItem(fLayer,0));

      //Ignoring the input layer for models built using Keras Functional API
      if(fLayerType == "'InputLayer'")
      continue;

      fAttributes=PyList_GetItem(fLayer,1);
      fInputs=PyList_GetItem(fLayer,2);
      fOutputs=PyList_GetItem(fLayer,3);
      fLayerDType = ConvertStringToType(PyStringAsString(PyDict_GetItemString(fAttributes,"dtype")));

      switch(INTERNAL::Type.find(fLayerType)->second){

         case LayerType::DENSE : {

            fLayerName       = PyStringAsString(PyDict_GetItemString(fAttributes,"name"));
            fLayerInput      = PyStringAsString(PyObject_GetAttrString(fInputs,"name"));
            fLayerOutput     = PyStringAsString(PyObject_GetAttrString(fOutputs,"name"));
            fLayerActivation = PyStringAsString(PyDict_GetItemString(fAttributes,"activation"));

            fWeightsNames = PyList_GetItem(fLayer,4);
            std::string kernel = PyStringAsString(PyList_GetItem(fWeightsNames,0));
            std::string bias   = PyStringAsString(PyList_GetItem(fWeightsNames,1));

                  if(fLayerActivation != "'linear'"){
                     rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Gemm(fLayerInput,fLayerName+"Gemm",kernel,bias,fLayerDType)));

                     if(INTERNAL::ActivationType.find(fLayerActivation)==INTERNAL::ActivationType.end())
                       throw std::runtime_error("Type error: Layer activation type "+fLayerActivation+" not yet registered in TMVA SOFIE");

                     switch(INTERNAL::ActivationType.find(fLayerActivation)->second){
                        case LayerType::RELU: {
                           rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Relu(fLayerName+"Gemm",fLayerOutput,fLayerDType)));
                           break;
                        }
                        default: throw std::runtime_error("Activation error: TMVA SOFIE does not yet suppport Activation type"+fLayerActivation);
                        }
                        }
                  else{
                     rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Gemm(fLayerInput,fLayerOutput,kernel,bias,fLayerDType)));
                  }

                  break;
               }

         case LayerType::ACTIVATION: {
            fLayerActivation = PyStringAsString(PyDict_GetItemString(fAttributes,"activation"));
            fLayerInput      = PyStringAsString(PyObject_GetAttrString(fInputs,"name"));
            fLayerOutput     = PyStringAsString(PyObject_GetAttrString(fOutputs,"name"));

            switch(INTERNAL::ActivationType.find(fLayerActivation)->second){
               case LayerType::RELU: {
                  rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Relu(fLayerInput,fLayerOutput,fLayerDType)));
                  break;
                  }
               default: throw std::runtime_error("Activation error: TMVA SOFIE does not yet suppport Activation type"+fLayerActivation);
               }
               break;
         }

         case LayerType::RELU: {
            fLayerInput  = PyStringAsString(PyObject_GetAttrString(fInputs,"name"));
            fLayerOutput = PyStringAsString(PyObject_GetAttrString(fOutputs,"name"));
            rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Relu(fLayerInput,fLayerOutput,fLayerDType)));
            break;
            }

         case LayerType::TRANSPOSE: {
            fLayerInput  = PyStringAsString(PyObject_GetAttrString(fInputs,"name"));
            fLayerOutput = PyStringAsString(PyObject_GetAttrString(fOutputs,"name"));
            PyObject* fAttributePermute=PyDict_GetItemString(fAttributes,"dims");
            std::vector<int_t>fPermuteDims;
            for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(fAttributePermute);++tupleIter){

               fPermuteDims.push_back((int_t)PyLong_AsLong(PyTuple_GetItem(fAttributePermute,tupleIter)));
            }
            rmodel.AddOperator(std::move(INTERNAL::make_ROperator_Transpose(fLayerInput,fLayerOutput,fPermuteDims,fLayerDType)));
            break;
            }
         default: throw std::runtime_error("Layer error: TMVA SOFIE does not yet suppport layer type"+fLayerType);
         }
         }


   //Extracting model's weights
   //For every initialized tensor, weightProp will have its name and dtype in string
   //and value in numpy array
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

         std::vector<size_t>fInputShape=GetShapeFromTuple(fPInputShapes);
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

         std::vector<size_t>fInputShape=GetShapeFromTuple(fInputShapeTuple);
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
