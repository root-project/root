// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDL                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Deep Neural Network Method                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TFormula.h"
#include "TString.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodDL.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/DLMinimizers.h"

REGISTER_METHOD(DL)
ClassImp(TMVA::MethodDL);

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA {

////////////////////////////////////////////////////////////////////////////////
TString fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key)
{
   key.ToUpper();
   std::map<TString, TString>::const_iterator it = keyValueMap.find(key);
   if (it == keyValueMap.end()) {
      return TString("");
   }
   return it->second;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
T fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, T defaultValue);

////////////////////////////////////////////////////////////////////////////////
template <>
int fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, int defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atoi();
}

////////////////////////////////////////////////////////////////////////////////
template <>
double fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, double defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atof();
}

////////////////////////////////////////////////////////////////////////////////
template <>
TString fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, TString defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value;
}

////////////////////////////////////////////////////////////////////////////////
template <>
bool fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key, bool defaultValue)
{
   TString value(fetchValueTmp(keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }

   value.ToUpper();
   if (value == "TRUE" || value == "T" || value == "1") {
      return true;
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////////
template <>
std::vector<double> fetchValueTmp(const std::map<TString, TString> &keyValueMap, TString key,
                                  std::vector<double> defaultValue)
{
   TString parseString(fetchValueTmp(keyValueMap, key));
   if (parseString == "") {
      return defaultValue;
   }

   parseString.ToUpper();
   std::vector<double> values;

   const TString tokenDelim("+");
   TObjArray *tokenStrings = parseString.Tokenize(tokenDelim);
   TIter nextToken(tokenStrings);
   TObjString *tokenString = (TObjString *)nextToken();
   for (; tokenString != NULL; tokenString = (TObjString *)nextToken()) {
      std::stringstream sstr;
      double currentValue;
      sstr << tokenString->GetString().Data();
      sstr >> currentValue;
      values.push_back(currentValue);
   }
   return values;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::DeclareOptions()
{
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ProcessOptions()
{
}

////////////////////////////////////////////////////////////////////////////////
/// default initializations
void MethodDL::Init()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Create a deep net based on the layout string
template <typename Architecture_t, typename Layer_t>
void MethodDL::CreateDeepNet(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                             std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets)
{
   // Layer specification, layer details
   const TString layerDelimiter(",");
   const TString subDelimiter("|");

   TString layoutString = this->GetLayoutString();

   // Split layers
   TObjArray *layerStrings = layoutString.Tokenize(layerDelimiter);
   TIter nextLayer(layerStrings);
   TObjString *layerString = (TObjString *)nextLayer();

   for (; layerString != nullptr; layerString = (TObjString *)nextLayer()) {
      // Split layer details
      TObjArray *subStrings = layerString->GetString().Tokenize(subDelimiter);
      TIter nextToken(subStrings);
      TObjString *token = (TObjString *)nextToken();

      // Determine the type of the layer
      TString strLayerType = token->GetString();

      if (strLayerType == "DENSE") {
         ParseDenseLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "CONV") {
         ParseConvLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "MAXPOOL") {
         ParseMaxPoolLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "RESHAPE") {
         ParseReshapeLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "RNN") {
         ParseRnnLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      } else if (strLayerType == "LSTM") {
         ParseLstmLayer(deepNet, nets, layerString->GetString(), subDelimiter);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate dense layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseDenseLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                               std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                               TString delim)
{
   int width = 0;
   EActivationFunction activationFunction = EActivationFunction::kTanh;

   // not sure about this
   const size_t inputSize = GetNvar();

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   // jump the first token
   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // number of nodes
      {
         // not sure
         TString strNumNodes(token->GetString());
         TString strN("x");
         strNumNodes.ReplaceAll("N", strN);
         strNumNodes.ReplaceAll("n", strN);
         TFormula fml("tmp", strNumNodes);
         width = fml.Eval(inputSize);
      } break;
      case 2: // actiovation function
      {
         TString strActFnc(token->GetString());
         if (strActFnc == "RELU") {
            activationFunction = DNN::EActivationFunction::kRelu;
         } else if (strActFnc == "TANH") {
            activationFunction = DNN::EActivationFunction::kTanh;
         } else if (strActFnc == "SYMMRELU") {
            activationFunction = DNN::EActivationFunction::kSymmRelu;
         } else if (strActFnc == "SOFTSIGN") {
            activationFunction = DNN::EActivationFunction::kSoftSign;
         } else if (strActFnc == "SIGMOID") {
            activationFunction = DNN::EActivationFunction::kSigmoid;
         } else if (strActFnc == "LINEAR") {
            activationFunction = DNN::EActivationFunction::kIdentity;
         } else if (strActFnc == "GAUSS") {
            activationFunction = DNN::EActivationFunction::kGauss;
         }
      } break;
      }
      ++idxToken;
   }

   // Add the dense layer, initialize the weights and biases and copy
   TDenseLayer<Architecture_t> *denseLayer = deepNet.AddDenseLayer(width, activationFunction);
   denseLayer->Initialize();
   TDenseLayer<Architecture_t> *copyDenseLayer = new TDenseLayer<Architecture_t>(*denseLayer);

   // add the copy to all slave nets
   for (size_t i = 0; i < nets.size(); i++) {
      nets[i].AddDenseLayer(copyDenseLayer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate convolutional layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseConvLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                              std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                              TString delim)
{
   int depth = 0;
   int fltHeight = 0;
   int fltWidth = 0;
   int strideRows = 0;
   int strideCols = 0;
   int zeroPadHeight = 0;
   int zeroPadWidth = 0;
   EActivationFunction activationFunction = EActivationFunction::kTanh;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // depth or width
      {
         TString strDepth(token->GetString());
         depth = strDepth.Atoi();
      } break;
      case 2: // filter height
      {
         TString strFltHeight(token->GetString());
         fltHeight = strFltHeight.Atoi();
      } break;
      case 3: // filter width
      {
         TString strFltWidth(token->GetString());
         fltWidth = strFltWidth.Atoi();
      } break;
      case 4: // stride in rows
      {
         TString strStrideRows(token->GetString());
         strideRows = strStrideRows.Atoi();
      } break;
      case 5: // stride in cols
      {
         TString strStrideCols(token->GetString());
         strideCols = strStrideCols.Atoi();
      } break;
      case 6: // zero padding height
      {
         TString strZeroPadHeight(token->GetString());
         zeroPadHeight = strZeroPadHeight.Atoi();
      } break;
      case 7: // zero padding width
      {
         TString strZeroPadWidth(token->GetString());
         zeroPadWidth = strZeroPadWidth.Atoi();
      } break;
      case 8: // activation function
      {
         TString strActFnc(token->GetString());
         if (strActFnc == "RELU") {
            activationFunction = DNN::EActivationFunction::kRelu;
         } else if (strActFnc == "TANH") {
            activationFunction = DNN::EActivationFunction::kTanh;
         } else if (strActFnc == "SYMMRELU") {
            activationFunction = DNN::EActivationFunction::kSymmRelu;
         } else if (strActFnc == "SOFTSIGN") {
            activationFunction = DNN::EActivationFunction::kSoftSign;
         } else if (strActFnc == "SIGMOID") {
            activationFunction = DNN::EActivationFunction::kSigmoid;
         } else if (strActFnc == "LINEAR") {
            activationFunction = DNN::EActivationFunction::kIdentity;
         } else if (strActFnc == "GAUSS") {
            activationFunction = DNN::EActivationFunction::kGauss;
         }
      } break;
      }
      ++idxToken;
   }

   // Add the convolutional layer, initialize the weights and biases and copy
   TConvLayer<Architecture_t> *convLayer = deepNet.AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                                                                zeroPadHeight, zeroPadWidth, activationFunction);
   convLayer->Initialize();
   TConvLayer<Architecture_t> *copyConvLayer = new TConvLayer<Architecture_t>(*convLayer);

   // add the copy to all slave nets
   for (size_t i = 0; i < nets.size(); i++) {
      nets[i].AddConvLayer(copyConvLayer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate max pool layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseMaxPoolLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                                 std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                                 TString delim)
{

   int frameHeight = 0;
   int frameWidth = 0;
   int strideRows = 0;
   int strideCols = 0;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // frame height
      {
         TString strFrmHeight(token->GetString());
         frameHeight = strFrmHeight.Atoi();
      } break;
      case 2: // frame width
      {
         TString strFrmWidth(token->GetString());
         frameWidth = strFrmWidth.Atoi();
      } break;
      case 3: // stride in rows
      {
         TString strStrideRows(token->GetString());
         strideRows = strStrideRows.Atoi();
      } break;
      case 4: // stride in cols
      {
         TString strStrideCols(token->GetString());
         strideCols = strStrideCols.Atoi();
      } break;
      }
      ++idxToken;
   }

   // Add the Max pooling layer
   TMaxPoolLayer<Architecture_t> *maxPoolLayer =
      deepNet.AddMaxPoolLayer(frameHeight, frameWidth, strideRows, strideCols);
   TMaxPoolLayer<Architecture_t> *copyMaxPoolLayer = new TMaxPoolLayer<Architecture_t>(*maxPoolLayer);

   // add the copy to all slave nets
   for (size_t i = 0; i < nets.size(); i++) {
      nets[i].AddMaxPoolLayer(copyMaxPoolLayer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate reshape layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseReshapeLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                                 std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                                 TString delim)
{
   int height = 0;
   int width = 0;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: // height
      {
         TString strHeight(token->GetString());
         height = strHeight.Atoi();
      } break;
      case 2: // width
      {
         TString strWidth(token->GetString());
         width = strWidth.Atoi();
      } break;
      }
      ++idxToken;
   }

   // Add the reshape layer
   TReshapeLayer<Architecture_t> *reshapeLayer = deepNet.AddReshapeLayer(height, width);
   TReshapeLayer<Architecture_t> *copyReshapeLayer = new TReshapeLayer<Architecture_t>(*reshapeLayer);

   // add the copy to all slave nets
   for (size_t i = 0; i < nets.size(); i++) {
      nets[i].AddReshapeLayer(copyReshapeLayer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate rnn layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseRnnLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                             std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                             TString delim)
{
   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      }
      ++idxToken;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pases the layer string and creates the appropriate lstm layer
template <typename Architecture_t, typename Layer_t>
void MethodDL::ParseLstmLayer(DNN::TDeepNet<Architecture_t, Layer_t> &deepNet,
                              std::vector<DNN::TDeepNet<Architecture_t, Layer_t>> &nets, TString layerString,
                              TString delim)
{
   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      }
      ++idxToken;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.
MethodDL::MethodDL(const TString &jobName, const TString &methodTitle, DataSetInfo &theData, const TString &theOption)
   : MethodBase(jobName, Types::kDL, methodTitle, theData, theOption), fWeightInitialization(), fOutputFunction(),
     fLossFunction(), fLayoutString(), fErrorStrategy(), fTrainingStrategyString(), fWeightInitializationString(),
     fArchitectureString(), fResume(false), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a weight file.
MethodDL::MethodDL(DataSetInfo &theData, const TString &theWeightFile)
   : MethodBase(Types::kDL, theData, theWeightFile), fWeightInitialization(), fOutputFunction(), fLossFunction(),
     fLayoutString(), fErrorStrategy(), fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
MethodDL::~MethodDL()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse key value pairs in blocks -> return vector of blocks with map of key value pairs.
auto MethodDL::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim) -> KeyValueVector_t
{
}

////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the CNN
Bool_t MethodDL::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::Train()
{
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::TrainGpu()
{
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::TrainCpu()
{
}

////////////////////////////////////////////////////////////////////////////////
Double_t MethodDL::GetMvaValue(Double_t * /*errLower*/, Double_t * /*errUpper*/)
{
   // TO DO
   return 0.0;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::AddWeightsXMLTo(void *parent) const
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ReadWeightsFromXML(void *rootXML)
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ReadWeightsFromStream(std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const Ranking *TMVA::MethodDL::CreateRanking()
{
   // TO DO
   return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::GetHelpMessage() const
{
   // TO DO
}

} // namespace TMVA