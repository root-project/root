// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta 2021

/**********************************************************************************
 * Project : TMVA - a Root-integrated toolkit for multivariate data analysis      *
 * Package : TMVA                                                                 *
 * Function: TMVA::Experimental::SOFIE::PyKeras::Parse                            *
 *                                                                                *
 * Description:                                                                   *
 *      Parser function for translating Keras .h5 model to RModel object          *
 *                                                                                *
 * Example Usage:                                                                 *
 * ~~~ {.cpp}                                                                     *
 * using TMVA::Experimental::SOFIE;                                               *
 * RModel model = PyKeras::Parse("trained_model_dense.h5");                       *
 * ~~~                                                                            *
 *                                                                                *
 **********************************************************************************/

#include "TMVA/RModelParser_Keras.h"

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyKeras{

// Referencing Python utility functions present in PyMethodBase
static void(& PyRunString)(TString, PyObject*, PyObject*) = PyMethodBase::PyRunString;
static const char*(& PyStringAsString)(PyObject*) = PyMethodBase::PyStringAsString;
static std::vector<size_t>(& GetDataFromTuple)(PyObject*) = PyMethodBase::GetDataFromTuple;
static PyObject*(& GetValueFromDict)(PyObject*, const char*) = PyMethodBase::GetValueFromDict;

namespace INTERNAL{

// For adding Keras layer into RModel object
void AddKerasLayer(RModel &rmodel, PyObject *fLayer);

// Declaring Internal Functions for Keras layers which don't have activation as an additional attribute
std::unique_ptr<ROperator> MakeKerasActivation(PyObject *fLayer);   // For instantiating ROperator for Keras Activation Layer
std::unique_ptr<ROperator> MakeKerasReLU(PyObject *fLayer);         // For instantiating ROperator for Keras ReLU layer
std::unique_ptr<ROperator> MakeKerasSelu(PyObject *fLayer);         // For instantiating ROperator for Keras Selu layer
std::unique_ptr<ROperator> MakeKerasSigmoid(PyObject *fLayer);      // For instantiating ROperator for Keras Sigmoid layer
std::unique_ptr<ROperator> MakeKerasSwish(PyObject *fLayer);        // For instantiating ROperator for Keras Swish layer
std::unique_ptr<ROperator> MakeKerasPermute(PyObject *fLayer);      // For instantiating ROperator for Keras Permute Layer
std::unique_ptr<ROperator> MakeKerasBatchNorm(PyObject *fLayer);    // For instantiating ROperator for Keras Batch Normalization Layer
std::unique_ptr<ROperator> MakeKerasReshape(PyObject *fLayer);      // For instantiating ROperator for Keras Reshape Layer
std::unique_ptr<ROperator> MakeKerasConcat(PyObject *fLayer);       // For instantiating ROperator for Keras Concat Layer
std::unique_ptr<ROperator> MakeKerasBinary(PyObject *fLayer);       // For instantiating ROperator for Keras binary operations: Add, Subtract & Multiply.
std::unique_ptr<ROperator> MakeKerasSoftmax(PyObject *fLayer);      // For instantiating ROperator for Keras Softmax Layer
std::unique_ptr<ROperator> MakeKerasTanh(PyObject *fLayer);         // For instantiating ROperator for Keras Tanh Layer
std::unique_ptr<ROperator> MakeKerasLeakyRelu(PyObject *fLayer);    // For instantiating ROperator for Keras LeakyRelu Layer
std::unique_ptr<ROperator> MakeKerasIdentity(PyObject *fLayer);     // For instantiating ROperator for Keras Identity Layer


// Declaring Internal function for Keras layers which have additional activation attribute
std::unique_ptr<ROperator> MakeKerasDense(PyObject *fLayer);        // For instantiating ROperator for Keras Dense Layer
std::unique_ptr<ROperator> MakeKerasConv(PyObject *fLayer);         // For instantiating ROperator for Keras Conv Layer

// For mapping Keras layer with the preparatory functions for ROperators
using KerasMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject *fLayer)>;
using KerasMethodMapWithActivation = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject *fLayer)>;

const KerasMethodMap mapKerasLayer = {
   {"Activation", &MakeKerasActivation},
   {"Permute", &MakeKerasPermute},
   {"BatchNormalization", &MakeKerasBatchNorm},
   {"Reshape", &MakeKerasReshape},
   {"Concatenate", &MakeKerasConcat},
   {"swish", &MakeKerasSwish},
   {"Add", &MakeKerasBinary},
   {"Subtract", &MakeKerasBinary},
   {"Multiply", &MakeKerasBinary},
   {"Softmax", &MakeKerasSoftmax},
   {"tanh", &MakeKerasTanh},
   {"LeakyReLU", &MakeKerasLeakyRelu},
   {"Identity",  &MakeKerasIdentity},
   {"Dropout",  &MakeKerasIdentity},

   // For activation layers
   {"ReLU", &MakeKerasReLU},

   // For layers with activation attributes
   {"relu", &MakeKerasReLU},
   {"selu", &MakeKerasSelu},
   {"sigmoid", &MakeKerasSigmoid},
   {"softmax", &MakeKerasSoftmax}
};

const KerasMethodMapWithActivation mapKerasLayerWithActivation = {
   {"Dense", &MakeKerasDense},
   {"Conv2D", &MakeKerasConv},
   };


//////////////////////////////////////////////////////////////////////////////////
/// \brief Adds equivalent ROperator with respect to Keras model layer
///        into the referenced RModel object
///
/// \param[in] rmodel RModel object
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \param[out] RModel object with the added ROperator
///
/// Function adds equivalent ROperator into the referenced RModel object.
/// Keras models can have layers like Dense and Conv which have activation
/// function as an attribute. Function first searches if layer object is among
/// the ones which don't have activation attribute and then calls the respective
/// preparation function to get the ROperator object, which is then added
/// into the RModel object. If passed layer is among the ones which may have activation
/// attribute, then it checks for the activation attribute, if present then first adds
/// the primary operator into the RModel object, and then adds the operator for the
/// activation function with appropriate changes in the names of input and output
/// tensors for both of them.
/// Example of such layers is the Dense Layer. For a dense layer with input tensor name
/// dense2BiasAdd0 and output tensor name dense3Relu0 with relu as activation attribute
/// will be transformed into a ROperator_Gemm with input tensor name dense2BiasAdd0
/// & output tensor name dense3Dense (layerName+layerType), and a subsequent
/// ROperator_Relu with input tensor name as dense3Dense and output tensor name
/// as dense3Relu0.
///
/// For developing new preparatory functions for supporting Keras layers in future,
/// all one needs is to extract the required properties and attributes from the fLayer
/// dictionary which contains all the information about any Keras layer and after
/// any required transformations, these are passed for instantiating the ROperator
/// object.
///
/// The fLayer dictionary which holds all the information about a Keras layer has
/// following structure:-
///
///     dict fLayer { 'layerType'       : Type of the Keras layer
///                   'layerAttributes' : Attributes of the keras layer as returned by layer.get_config()
///                   'layerInput'      : List of names of input tensors
///                   'layerOutput'     : List of names of output tensors
///                   'layerDType'      : Data-type of the Keras layer
///                   'layerWeight'     : List of weight tensor names of Keras layers
///                 }
void AddKerasLayer(RModel& rmodel, PyObject* fLayer){
   std::string fLayerType = PyStringAsString(GetValueFromDict(fLayer,"layerType"));

   if(fLayerType == "Reshape"){
      PyObject* fAttributes=GetValueFromDict(fLayer,"layerAttributes");
      std::string fLayerName = PyStringAsString(GetValueFromDict(fAttributes,"_name"));
      PyObject* fPTargetShape = GetValueFromDict(fAttributes,"target_shape");
      std::vector<size_t>fTargetShape = GetDataFromTuple(fPTargetShape);
      std::shared_ptr<void> fData(malloc(fTargetShape.size() * sizeof(int64_t)), free);
      std::copy(fTargetShape.begin(),fTargetShape.end(),(int64_t*)fData.get());
      rmodel.AddInitializedTensor(fLayerName+"ReshapeAxes",ETensorType::INT64,{fTargetShape.size()},fData);
   }

   //For layers without additional activation attribute
   auto findLayer = mapKerasLayer.find(fLayerType);
   if(findLayer != mapKerasLayer.end()){
      rmodel.AddOperator((findLayer->second)(fLayer));
      return;
   }

   //For layers like Dense & Conv which has additional activation attribute
   else if(mapKerasLayerWithActivation.find(fLayerType) != mapKerasLayerWithActivation.end()){
      findLayer = mapKerasLayerWithActivation.find(fLayerType);
      PyObject* fAttributes=GetValueFromDict(fLayer,"layerAttributes");

      std::string fLayerName = PyStringAsString(GetValueFromDict(fAttributes,"_name"));

      PyObject* fPActivation = GetValueFromDict(fAttributes,"activation");
      std::string fLayerActivation = PyStringAsString(PyObject_GetAttrString(fPActivation,"__name__"));

      if(fLayerActivation == "selu" || fLayerActivation == "sigmoid")
         rmodel.AddNeededStdLib("cmath");


      //Checking if additional attribute exixts
      if(fLayerActivation != "linear"){
         PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");
         PyObject* fInputs = GetValueFromDict(fLayer,"layerInput");
         std::string fActivationLayerOutput = PyStringAsString(PyList_GetItem(fOutputs,0));

         if(fLayerType == "Conv2D"){
            std::unique_ptr<ROperator> op_pre_transpose;
            op_pre_transpose.reset(new ROperator_Transpose<float>({0,3,1,2}, PyStringAsString(PyList_GetItem(fInputs,0)), fLayerName+"PreTrans"));
            rmodel.AddOperator(std::move(op_pre_transpose));

            PyList_SetItem(fInputs,0,PyUnicode_FromString((fLayerName+"PreTrans").c_str()));
            PyDict_SetItemString(fLayer,"layerInput",fInputs);
         }

         // Making changes in the names of the input and output tensor names
         PyList_SetItem(fOutputs,0,PyUnicode_FromString((fLayerName+fLayerType).c_str()));
         PyDict_SetItemString(fLayer,"layerOutput",fOutputs);
         rmodel.AddOperator((findLayer->second)(fLayer));

         std::string fActivationLayerInput = fLayerName+fLayerType;
         if(fLayerType == "Conv2D"){
            std::unique_ptr<ROperator> op_post_transpose;
            op_post_transpose.reset(new ROperator_Transpose<float>({0,2,3,1}, fLayerName+fLayerType, fLayerName+"PostTrans"));
            rmodel.AddOperator(std::move(op_post_transpose));
            fActivationLayerInput = fLayerName+"PostTrans";
         }

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

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Dense Layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For Keras's Dense layer, the names of the input tensor, output tensor, and
/// weight tensors are extracted, and then are passed to instantiate a
/// ROperator_Gemm object using the required attributes.
std::unique_ptr<ROperator> MakeKerasDense(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");
      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));

      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      // Extracting names of weight tensors
      // The names of Kernel weights and bias weights are found in the list
      // of weight tensors from fLayer.
      PyObject* fWeightNames  = GetValueFromDict(fLayer,"layerWeight");
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



//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Conv Layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For Keras's Conv layer, the names of the input tensor, output tensor, and
/// weight tensors are extracted, along with attributes like dilation_rate,
/// groups, kernel size, padding, strides. Padding attribute is then
/// computed for ROperator depending on Keras' attribute parameter.
std::unique_ptr<ROperator> MakeKerasConv(PyObject* fLayer){
      PyObject* fAttributes = GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");
      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));

      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      // Extracting names of weight tensors
      // The names of Kernel weights and bias weights are found in the list
      // of weight tensors from fLayer.
      PyObject* fWeightNames  = GetValueFromDict(fLayer,"layerWeight");
      std::string fKernelName = PyStringAsString(PyList_GetItem(fWeightNames,0));
      std::string fBiasName   = PyStringAsString(PyList_GetItem(fWeightNames,1));

      // Extracting the Conv Node Attributes
      PyObject* fDilations       = GetValueFromDict(fAttributes,"dilation_rate");
      PyObject* fGroup           = GetValueFromDict(fAttributes,"groups");
      PyObject* fKernelShape     = GetValueFromDict(fAttributes,"kernel_size");
      PyObject* fPads            = GetValueFromDict(fAttributes,"padding");
      PyObject* fStrides         = GetValueFromDict(fAttributes,"strides");

      std::vector<size_t> fAttrDilations = GetDataFromTuple(fDilations);


      size_t fAttrGroup = PyLong_AsLong(fGroup);
      std::vector<size_t> fAttrKernelShape = GetDataFromTuple(fKernelShape);
      std::vector<size_t> fAttrStrides     = GetDataFromTuple(fStrides);
      std::string fAttrAutopad;
      std::vector<size_t>fAttrPads;

      //Seting the layer padding
      std::string fKerasPadding = PyStringAsString(fPads);
      if(fKerasPadding == "valid"){
         fAttrAutopad = "VALID";
      }
      else if(fKerasPadding == "same"){
         fAttrAutopad="NOTSET";
         PyObject* fInputShape  = GetValueFromDict(fAttributes,"_batch_input_shape");
         long inputHeight = PyLong_AsLong(PyTuple_GetItem(fInputShape,1));
         long inputWidth = PyLong_AsLong(PyTuple_GetItem(fInputShape,2));

         long outputHeight = std::ceil(float(inputHeight) / float(fAttrStrides[0]));
         long outputWidth  = std::ceil(float(inputWidth) / float(fAttrStrides[1]));

         long padding_height = std::max(long((outputHeight - 1) * fAttrStrides[0] + fAttrKernelShape[0] - inputHeight),0L);
         long padding_width = std::max(long((outputWidth - 1) * fAttrStrides[1] + fAttrKernelShape[1] - inputWidth),0L);

         size_t padding_top = std::floor(padding_height/2);
         size_t padding_bottom   = padding_height - padding_top;
         size_t padding_left = std::floor(padding_width/2);
         size_t padding_right   = padding_width - padding_left;
         fAttrPads = {padding_top,padding_bottom,padding_left,padding_right};
      }
      else{
         throw std::runtime_error("TMVA::SOFIE - RModel Keras Parser doesn't yet supports Convolution layer with padding " + fKerasPadding);
      }

      std::unique_ptr<ROperator> op;

      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Conv<float>(fAttrAutopad, fAttrDilations, fAttrGroup, fAttrKernelShape, fAttrPads, fAttrStrides, fLayerInputName, fKernelName, fBiasName, fLayerOutputName));
         break;

         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Conv does not yet support input type " + fLayerDType);
         }
         return op;
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras activation layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For Keras's keras.layers.Activation layer, the activation attribute is
/// extracted and appropriate function for adding the function is called.
std::unique_ptr<ROperator> MakeKerasActivation(PyObject* fLayer){
      PyObject* fAttributes=GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fPActivation = GetValueFromDict(fAttributes,"activation");
      std::string fLayerActivation = PyStringAsString(PyObject_GetAttrString(fPActivation,"__name__"));

      auto findLayer = mapKerasLayer.find(fLayerActivation);
      if(findLayer == mapKerasLayer.end()){
         throw std::runtime_error("TMVA::SOFIE - Parsing Keras Activation layer " + fLayerActivation + " is not yet supported");
      }
      return (findLayer->second)(fLayer);
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras ReLU activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Relu object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasReLU(PyObject* fLayer)
{
      PyObject* fInputs=GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs=GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
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


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Selu activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Selu object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasSelu(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Selu<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Selu does not yet support input type " + fLayerDType);
         }
   return op;
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Sigmoid activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Sigmoid object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasSigmoid(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Sigmoid<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Softmax activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Softmax object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasSoftmax(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Softmax<float>(/*default axis is -1*/-1,fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Leaky Relu activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_LeakyRelu object, the names of
/// input & output tensors, the data-type and the alpha attribute of the layer
/// are extracted.
std::unique_ptr<ROperator> MakeKerasLeakyRelu(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");
      PyObject* fAttributes=GetValueFromDict(fLayer,"layerAttributes");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));
      float fAlpha = (float)PyFloat_AsDouble(GetValueFromDict(fAttributes,"alpha"));
      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_LeakyRelu<float>(fAlpha, fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Tanh activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Tanh object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasTanh(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Tanh<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Tanh does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Swish activation
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Swish object, the names of
/// input & output tensors and the data-type of the layer are extracted.
std::unique_ptr<ROperator> MakeKerasSwish(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Swish<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Swish does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Permute layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// The Permute layer in Keras has an equivalent Tranpose operator in ONNX.
/// For adding a Transpose operator, the permute dimensions are found, if they
/// exist are passed in instantiating the ROperator, else default values are used.
std::unique_ptr<ROperator> MakeKerasPermute(PyObject* fLayer)
{
      // Extracting required layer information
      PyObject* fAttributes=GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fInputs=GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs=GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName      = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName     = PyStringAsString(PyList_GetItem(fOutputs,0));

      // Extracting the permute dimensions present in Attributes of the Keras layer
      PyObject* fAttributePermute = GetValueFromDict(fAttributes,"dims");
      std::vector<int_t>fPermuteDims;

      // Building vector of permute dimensions from the Tuple object.
      for(Py_ssize_t tupleIter=0;tupleIter<PyTuple_Size(fAttributePermute);++tupleIter){

         fPermuteDims.push_back((int_t)PyLong_AsLong(PyTuple_GetItem(fAttributePermute,tupleIter)));
      }
      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:{

            // Adding the permute dimensions if present, else are avoided to use default values.
            if (!fPermuteDims.empty()){
               op.reset(new ROperator_Transpose<float>(fPermuteDims, fLayerInputName, fLayerOutputName));
               }
            else{
               op.reset(new ROperator_Transpose<float> (fLayerInputName, fLayerOutputName));
               }
         break;
         }
         default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + fLayerDType);
            }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras BatchNorm layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
std::unique_ptr<ROperator> MakeKerasBatchNorm(PyObject* fLayer)
{
      // Extracting required layer information
      PyObject* fAttributes = GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");
      PyObject* fGamma =  GetValueFromDict(fAttributes,"gamma");
      PyObject* fBeta =  GetValueFromDict(fAttributes,"beta");
      PyObject* fMoving_Mean =  GetValueFromDict(fAttributes,"moving_mean");
      PyObject* fMoving_Var =  GetValueFromDict(fAttributes,"moving_variance");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fNX      = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fNY     = PyStringAsString(PyList_GetItem(fOutputs,0));
      std::string fNScale  = PyStringAsString(PyObject_GetAttrString(fGamma,"name"));
      std::string fNB  = PyStringAsString(PyObject_GetAttrString(fBeta,"name"));
      std::string fNMean  = PyStringAsString(PyObject_GetAttrString(fMoving_Mean,"name"));
      std::string fNVar  = PyStringAsString(PyObject_GetAttrString(fMoving_Var,"name"));
      float fEpsilon = (float)PyFloat_AsDouble(GetValueFromDict(fAttributes,"epsilon"));
      float fMomentum = (float)PyFloat_AsDouble(GetValueFromDict(fAttributes,"momentum"));

      std::unique_ptr<ROperator> op;
      op.reset(new ROperator_BatchNormalization<float>(fEpsilon, fMomentum, /* training mode */ 0, fNX, fNScale, fNB, fNMean, fNVar, fNY));
      return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Reshape layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
std::unique_ptr<ROperator> MakeKerasReshape(PyObject* fLayer)
{
      // Extracting required layer information
      PyObject* fAttributes = GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerName = PyStringAsString(GetValueFromDict(fAttributes,"_name"));

      ReshapeOpMode fOpMode = Reshape;
      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fNameData      = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fNameOutput    = PyStringAsString(PyList_GetItem(fOutputs,0));
      std::string fNameShape     = fLayerName + "ReshapeAxes";
      std::unique_ptr<ROperator> op;
      op.reset(new ROperator_Reshape(fOpMode, /*allow zero*/0, fNameData, fNameShape, fNameOutput));
      return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Concat layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
std::unique_ptr<ROperator> MakeKerasConcat(PyObject* fLayer)
{
      PyObject* fAttributes = GetValueFromDict(fLayer,"layerAttributes");
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::vector<std::string> inputs;
      for(Py_ssize_t i=0; i<PyList_Size(fInputs); ++i){
         inputs.emplace_back(PyStringAsString(PyList_GetItem(fInputs,i)));
      }
      std::string output  = PyStringAsString(PyList_GetItem(fOutputs,0));

      int axis = (int)PyLong_AsLong(GetValueFromDict(fAttributes,"axis"));
      std::unique_ptr<ROperator> op;
      op.reset(new ROperator_Concat<float>(inputs, axis, 0,  output));
      return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras binary operations like Add,
///        subtract, and multiply.
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_BasicBinary object, the names of
/// input & output tensors, the data-type of the layer and the operation type
/// are extracted.
std::unique_ptr<ROperator> MakeKerasBinary(PyObject* fLayer){
      PyObject* fInputs  = GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs = GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerType = PyStringAsString(GetValueFromDict(fLayer,"layerType"));
      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fX1  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fX2  = PyStringAsString(PyList_GetItem(fInputs,1));
      std::string fY   = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:{
            if(fLayerType == "Add")
               op.reset(new ROperator_BasicBinary<float, Add> (fX1, fX2, fY));
            else if(fLayerType == "Subtract")
               op.reset(new ROperator_BasicBinary<float, Sub> (fX1, fX2, fY));
            else
               op.reset(new ROperator_BasicBinary<float, Mul> (fX1, fX2, fY));
         break;
         }
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Sigmoid does not yet support input type " + fLayerDType);
         }
   return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator object for Keras Identity and Dropout Layer
///
/// \param[in] fLayer Python Keras layer as a Dictionary object
/// \return Unique pointer to ROperator object
///
/// Dropout will have no effect in inference, so instead an Identity operator
/// is added to mimic its presence in the Keras model
std::unique_ptr<ROperator> MakeKerasIdentity(PyObject* fLayer)
{
      PyObject* fInputs=GetValueFromDict(fLayer,"layerInput");
      PyObject* fOutputs=GetValueFromDict(fLayer,"layerOutput");

      std::string fLayerDType = PyStringAsString(GetValueFromDict(fLayer,"layerDType"));
      std::string fLayerInputName  = PyStringAsString(PyList_GetItem(fInputs,0));
      std::string fLayerOutputName = PyStringAsString(PyList_GetItem(fOutputs,0));

      std::unique_ptr<ROperator> op;
      switch(ConvertStringToType(fLayerDType)){
         case ETensorType::FLOAT:
         op.reset(new ROperator_Identity<float>(fLayerInputName, fLayerOutputName));
         break;
         default:
         throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Identity does not yet support input type " + fLayerDType);
         }
   return op;
}

}//INTERNAL


//////////////////////////////////////////////////////////////////////////////////
/// \param[in] filename file location of Keras .h5
/// \return Parsed RModel object
///
/// The `Parse()` function defined in `TMVA::Experimental::SOFIE::PyKeras` will
/// parse a trained Keras .h5 model into a RModel Object. After loading the model
/// in a Python Session, the included layers are extracted with properties
/// like Layer type, Attributes, Input tensor names, Output tensor names, data-type
/// and names of the weight/initialized tensors.
/// The extracted layers from the model are then passed into `AddKerasLayer()`
/// which prepares the specific ROperator and adds them into the RModel object.
/// The layers are also checked for adding any required routines for executing
/// the generated Inference code.
///
/// For adding the Initialized tensors into the RModel object, the weights are
/// extracted from the Keras model in the form of NumPy arrays, which are then
/// passed into `AddInitializedTensor()` after appropriate casting.
///
/// Input tensor infos are required to be added which will contain their names,
/// shapes and data-types. For keras models with single input tensors, the tensor
/// shape is returned as a Tuple object, whereas for multi-input models,
/// the tensor shape is returned as a List of Tuple object containing the shape
/// of the individual input tensors. SOFIE's RModel also requires that the Keras
/// models are initialized with Batch Size. The `GetDataFromTuple()` are called
/// on the Tuple objects, which then returns the shape vector required to call
/// the `AddInputTensorInfo()`.
///
/// For adding the Output Tensor infos, only the names of the model's output
/// tensors are extracted and are then passed into `AddOutputTensorNameList()`.
///
/// Provide optionally a batch size that can be used to overwrite the one given by the
/// model. If a batch size is not given 1 is used if the model does not provide a batch size
///
/// Example Usage:
/// ~~~ {.cpp}
/// using TMVA::Experimental::SOFIE;
/// RModel model = PyKeras::Parse("trained_model_dense.h5");
/// ~~~
RModel Parse(std::string filename, int batch_size){

   char sep = '/';
   #ifdef _WIN32
   sep = '\\';
   #endif

   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos){
      filename_nodir = (filename.substr(isep+1, filename.length() - isep));
   }

   //Check on whether the Keras .h5 file exists
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

   // Extracting model information
   // For each layer: type,name,activation,dtype,input tensor's name,
   // output tensor's name, kernel's name, bias's name
   // None object is returned for if property doesn't belong to layer
   PyRunString("import tensorflow",fGlobalNS,fLocalNS);
   PyRunString("import tensorflow.keras as keras",fGlobalNS,fLocalNS);
   PyRunString("from tensorflow.keras.models import load_model",fGlobalNS,fLocalNS);
   PyRunString("print('TF/Keras Version: '+ tensorflow.__version__)",fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model=load_model('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString(TString::Format("model.load_weights('%s')",filename.c_str()),fGlobalNS,fLocalNS);
   PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
   PyRunString("modelData=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.layers)):\n"
               "  layer=model.get_layer(index=idx)\n"
               "  layerData={}\n"
               "  layerData['layerType']=layer.__class__.__name__\n"
               "  layerData['layerAttributes']=layer.__dict__\n"
               "  layerData['layerInput']=[x.name for x in layer.input] if isinstance(layer.input,list) else [layer.input.name]\n"
               "  layerData['layerOutput']=[x.name for x in layer.output] if isinstance(layer.output,list) else [layer.output.name]\n"
               "  layerData['layerDType']=layer.dtype\n"
               "  layerData['layerWeight']=[x.name for x in layer.weights]\n"
               "  modelData.append(layerData)",fGlobalNS,fLocalNS);


   PyObject* fPModel = GetValueFromDict(fLocalNS,"modelData");
   PyObject *fLayer;
   Py_ssize_t fModelSize = PyList_Size(fPModel);
   std::string fLayerType;

   // Traversing through all the layers and passing the Layer object to `AddKerasLayer()`
   // for adding the equivalent ROperators into the RModel object.
   for(Py_ssize_t fModelIterator=0;fModelIterator<fModelSize;++fModelIterator){
      fLayer     = PyList_GetItem(fPModel,fModelIterator);
      fLayerType = PyStringAsString(GetValueFromDict(fLayer,"layerType"));

      // Ignoring the input layer for models built using Keras Functional API
      if(fLayerType == "InputLayer")
         continue;

      // Adding any required routines depending on the Layer types for generating
      // inference code.
      else if(fLayerType == "Dense")
         rmodel.AddBlasRoutines({"Gemm", "Gemv"});
      else if (fLayerType == "BatchNormalization")
         rmodel.AddBlasRoutines({"Copy", "Axpy"});
      else if(fLayerType == "Conv1D" || fLayerType == "Conv2D" || fLayerType == "Conv3D")
         rmodel.AddBlasRoutines({"Gemm", "Axpy"});

      INTERNAL::AddKerasLayer(rmodel,fLayer);

   }

   //Extracting model's weights
   //For every initialized tensor, weightProp will have its name and dtype in string
   //and value in numpy array
   PyRunString("weight=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.get_weights())):\n"
               "  weightProp={}\n"
               "  weightProp['name']=model.weights[idx].name\n"
               "  weightProp['dtype']=(model.get_weights())[idx].dtype.name\n"
               "  weightProp['value']=(model.get_weights())[idx].transpose((3,2,0,1)).copy() if ('conv' in model.weights[idx].name and model.weights[idx].shape.ndims == 4) else (model.get_weights())[idx]\n"
               "  weight.append(weightProp)",fGlobalNS,fLocalNS);

   PyObject *fWeightTensor, *fPWeight;
   PyArrayObject *fWeightTensorValue;
   std::string fWeightName;
   ETensorType fWeightDType;
   fPWeight = GetValueFromDict(fLocalNS,"weight");
   std::vector<std::size_t> fWeightTensorShape;
   std::size_t fWeightTensorSize;

   // Traversing through all the Weight tensors
   for (Py_ssize_t weightIter = 0; weightIter < PyList_Size(fPWeight); weightIter++){
      fWeightTensor      = PyList_GetItem(fPWeight, weightIter);
      fWeightName        = PyStringAsString(GetValueFromDict(fWeightTensor,"name"));
      fWeightDType       = ConvertStringToType(PyStringAsString(GetValueFromDict(fWeightTensor,"dtype")));

      fWeightTensorValue = (PyArrayObject*)GetValueFromDict(fWeightTensor,"value");
      fWeightTensorSize=1;
      fWeightTensorShape.clear();

      // Building the shape vector and finding the tensor size
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


   // Extracting input tensor info
   // For every input tensor inputNames will have their names as string,
   // inputShapes will have their shape as Python Tuple, and inputTypes
   // will have their dtype as string
   PyRunString("inputNames=model.input_names",fGlobalNS,fLocalNS);
   PyRunString("inputShapes=model.input_shape if type(model.input_shape)==list else [model.input_shape]",fGlobalNS,fLocalNS);
   PyRunString("inputTypes=[]",fGlobalNS,fLocalNS);
   PyRunString("for idx in range(len(model.inputs)):\n"
               "  inputTypes.append(model.inputs[idx].dtype.__str__()[9:-2])",fGlobalNS,fLocalNS);

   PyObject* fPInputs       = GetValueFromDict(fLocalNS,"inputNames");
   PyObject* fPInputShapes  = GetValueFromDict(fLocalNS,"inputShapes");
   PyObject* fPInputTypes   = GetValueFromDict(fLocalNS,"inputTypes");

   std::string fInputName;
   ETensorType fInputDType;

   // For single input models, the model.input_shape will return a tuple
   // describing the input tensor shape. For multiple inputs models,
   // the model.input_shape will return a list of tuple, each describing
   // the input tensor shape.
   if(PyTuple_Check(fPInputShapes)){
      fInputName  = PyStringAsString(PyList_GetItem(fPInputs,0));
      fInputDType = ConvertStringToType(PyStringAsString(PyList_GetItem(fPInputTypes,0)));

      switch(fInputDType){

         case ETensorType::FLOAT : {

         // Getting the shape vector from the Tuple object
         std::vector<size_t>fInputShape = GetDataFromTuple(fPInputShapes);
         if (static_cast<int>(fInputShape[0]) <= 0){
            fInputShape[0] = std::max(batch_size,1);
            std::cout << "Model has not a defined batch size ";
            if (batch_size <=0) std::cout << " assume is 1 ";
            else std::cout << " use given value of " << batch_size;
            std::cout << " - input shape for tensor " << fInputName << " : "
                      << TMVA::Experimental::SOFIE::ConvertShapeToString(fInputShape) << std::endl;
         }
         rmodel.AddInputTensorInfo(fInputName, ETensorType::FLOAT, fInputShape);
         rmodel.AddInputTensorName(fInputName);
         break;
         }

         default:
         throw std::runtime_error("Type error: TMVA SOFIE does not yet support data type"+ConvertTypeToString(fInputDType));
      }

   }

   else{

      // Iterating through multiple input tensors
      for(Py_ssize_t inputIter = 0; inputIter < PyList_Size(fPInputs);++inputIter){

      fInputName  = PyStringAsString(PyList_GetItem(fPInputs,inputIter));
      fInputDType = ConvertStringToType(PyStringAsString(PyList_GetItem(fPInputTypes,inputIter)));

      switch(fInputDType){
         case ETensorType::FLOAT : {
         PyObject* fInputShapeTuple=PyList_GetItem(fPInputShapes,inputIter);

         std::vector<size_t>fInputShape = GetDataFromTuple(fInputShapeTuple);
         if (static_cast<int>(fInputShape[0]) <= 0){
            fInputShape[0] = std::max(batch_size,1);
            std::cout << "Model has not a defined batch size ";
            if (batch_size <=0) std::cout << " assume is 1 ";
            else std::cout << " use given value of " << batch_size;
            std::cout << " - input shape for tensor "
                      << fInputName << " : " << TMVA::Experimental::SOFIE::ConvertShapeToString(fInputShape) << std::endl;
         }
         rmodel.AddInputTensorInfo(fInputName, ETensorType::FLOAT, fInputShape);
         rmodel.AddInputTensorName(fInputName);
         break;
         }

         default:
         throw std::runtime_error("Type error: TMVA SOFIE does not yet support data type"+ConvertTypeToString(fInputDType));

      }
   }
   }


   // For adding OutputTensorInfos, the names of the output
   // tensors are extracted from the Keras model
   PyRunString("outputNames=[]",fGlobalNS,fLocalNS);
   PyRunString("for layerName in model.output_names:\n"
               "    outputNames.append(model.get_layer(layerName).output.name)",fGlobalNS,fLocalNS);
   PyObject* fPOutputs   = GetValueFromDict(fLocalNS,"outputNames");
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
