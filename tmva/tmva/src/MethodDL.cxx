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

#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
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
   // Set default values for all option strings

   DeclareOptionRef(fInputLayoutString = "0|0|0", "InputLayout", "The Layout of the input");

   DeclareOptionRef(fBatchLayoutString = "0|0|0", "BatchLayout", "The Layout of the batch");

   DeclareOptionRef(fLayoutString = "DENSE|(N+100)*2|SOFTSIGN,DENSE|0|LINEAR", "Layout", "Layout of the network.");

   DeclareOptionRef(fErrorStrategy = "CROSSENTROPY", "ErrorStrategy", "Loss function: Mean squared error (regression)"
                                                                      " or cross entropy (binary classification).");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));

   DeclareOptionRef(fWeightInitializationString = "XAVIER", "WeightInitialization", "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));

   DeclareOptionRef(fArchitectureString = "CPU", "Architecture", "Which architecture to perform the training on.");
   AddPreDefVal(TString("STANDARD"));
   AddPreDefVal(TString("CPU"));
   AddPreDefVal(TString("GPU"));
   AddPreDefVal(TString("OPENCL"));

   DeclareOptionRef(fTrainingStrategyString = "LearningRate=1e-1,"
                                              "Momentum=0.3,"
                                              "Repetitions=3,"
                                              "ConvergenceSteps=50,"
                                              "BatchSize=30,"
                                              "TestRepetitions=7,"
                                              "WeightDecay=0.0,"
                                              "Renormalize=L2,"
                                              "DropConfig=0.0,"
                                              "DropRepetitions=5|LearningRate=1e-4,"
                                              "Momentum=0.3,"
                                              "Repetitions=3,"
                                              "ConvergenceSteps=50,"
                                              "BatchSize=20,"
                                              "TestRepetitions=7,"
                                              "WeightDecay=0.001,"
                                              "Renormalize=L2,"
                                              "DropConfig=0.0+0.5+0.5,"
                                              "DropRepetitions=5,"
                                              "Multithreading=True",
                    "TrainingStrategy", "Defines the training strategies.");
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::ProcessOptions()
{
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO << "Will ignore negative events in training!" << Endl;
   }

   if (fArchitectureString == "STANDARD") {
      Log() << kERROR << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;
      Log() << kFATAL << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;
   }

   if (fArchitectureString == "OPENCL") {
      Log() << kERROR << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
      Log() << kFATAL << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
   }

   if (fArchitectureString == "GPU") {
#ifndef DNNCUDA // Included only if DNNCUDA flag is _not_ set.
      Log() << kERROR << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
      Log() << kFATAL << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
#endif // DNNCUDA
   }

   if (fArchitectureString == "CPU") {
#ifndef DNNCPU // Included only if DNNCPU flag is _not_ set.
      Log() << kERROR << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
      Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
#endif // DNNCPU
   }

   // Input Layout
   ParseInputLayout();
   ParseBatchLayout();

   // Loss function and output.
   fOutputFunction = EOutputFunction::kSigmoid;
   if (fAnalysisType == Types::kClassification) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fLossFunction = ELossFunction::kMeanSquaredError;
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fLossFunction = ELossFunction::kCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSigmoid;
   } else if (fAnalysisType == Types::kRegression) {
      if (fErrorStrategy != "SUMOFSQUARES") {
         Log() << kWARNING << "For regression only SUMOFSQUARES is a valid "
               << " neural net error function. Setting error function to "
               << " SUMOFSQUARES now." << Endl;
      }

      fLossFunction = ELossFunction::kMeanSquaredError;
      fOutputFunction = EOutputFunction::kIdentity;
   } else if (fAnalysisType == Types::kMulticlass) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fLossFunction = ELossFunction::kMeanSquaredError;
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fLossFunction = ELossFunction::kCrossEntropy;
      }
      if (fErrorStrategy == "MUTUALEXCLUSIVE") {
         fLossFunction = ELossFunction::kSoftmaxCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSoftmax;
   }

   // Initialization
   if (fWeightInitializationString == "XAVIER") {
      fWeightInitialization = DNN::EInitialization::kGauss;
   } else if (fWeightInitializationString == "XAVIERUNIFORM") {
      fWeightInitialization = DNN::EInitialization::kUniform;
   } else {
      fWeightInitialization = DNN::EInitialization::kGauss;
   }

   // Training settings.

   KeyValueVector_t strategyKeyValues = ParseKeyValueString(fTrainingStrategyString, TString("|"), TString(","));
   for (auto &block : strategyKeyValues) {
      TTrainingSettings settings;

      settings.convergenceSteps = fetchValueTmp(block, "ConvergenceSteps", 100);
      settings.batchSize = fetchValueTmp(block, "BatchSize", 30);
      settings.testInterval = fetchValueTmp(block, "TestRepetitions", 7);
      settings.weightDecay = fetchValueTmp(block, "WeightDecay", 0.0);
      settings.learningRate = fetchValueTmp(block, "LearningRate", 1e-5);
      settings.momentum = fetchValueTmp(block, "Momentum", 0.3);
      settings.dropoutProbabilities = fetchValueTmp(block, "DropConfig", std::vector<Double_t>());

      TString regularization = fetchValueTmp(block, "Regularization", TString("NONE"));
      if (regularization == "L1") {
         settings.regularization = DNN::ERegularization::kL1;
      } else if (regularization == "L2") {
         settings.regularization = DNN::ERegularization::kL2;
      }

      TString strMultithreading = fetchValueTmp(block, "Multithreading", TString("True"));

      if (strMultithreading.BeginsWith("T")) {
         settings.multithreading = true;
      } else {
         settings.multithreading = false;
      }

      fTrainingSettings.push_back(settings);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// default initializations
void MethodDL::Init()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the input layout
void MethodDL::ParseInputLayout()
{
   // Define the delimiter
   const TString delim("|");

   // Get the input layout string
   TString inputLayoutString = this->GetInputLayoutString();

   size_t depth = 0;
   size_t height = 0;
   size_t width = 0;

   // Split the input layout string
   TObjArray *inputDimStrings = inputLayoutString.Tokenize(delim);
   TIter nextInputDim(inputDimStrings);
   TObjString *inputDimString = (TObjString *)nextInputDim();
   int idxToken = 0;

   for (; inputDimString != nullptr; inputDimString = (TObjString *)nextInputDim()) {
      switch (idxToken) {
      case 0: // input depth
      {
         TString strDepth(inputDimString->GetString());
         depth = (size_t)strDepth.Atoi();
      } break;
      case 1: // input height
      {
         TString strHeight(inputDimString->GetString());
         height = (size_t)strHeight.Atoi();
      } break;
      case 2: // input width
      {
         TString strWidth(inputDimString->GetString());
         width = (size_t)strWidth.Atoi();
      } break;
      }
      ++idxToken;
   }

   this->SetInputDepth(depth);
   this->SetInputHeight(height);
   this->SetInputWidth(width);
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the input layout
void MethodDL::ParseBatchLayout()
{
   // Define the delimiter
   const TString delim("|");

   // Get the input layout string
   TString batchLayoutString = this->GetBatchLayoutString();

   size_t batchDepth = 0;
   size_t batchHeight = 0;
   size_t batchWidth = 0;

   // Split the input layout string
   TObjArray *batchDimStrings = batchLayoutString.Tokenize(delim);
   TIter nextBatchDim(batchDimStrings);
   TObjString *batchDimString = (TObjString *)nextBatchDim();
   int idxToken = 0;

   for (; batchDimString != nullptr; batchDimString = (TObjString *)nextBatchDim()) {
      switch (idxToken) {
      case 0: // input depth
      {
         TString strDepth(batchDimString->GetString());
         batchDepth = (size_t)strDepth.Atoi();
      } break;
      case 1: // input height
      {
         TString strHeight(batchDimString->GetString());
         batchHeight = (size_t)strHeight.Atoi();
      } break;
      case 2: // input width
      {
         TString strWidth(batchDimString->GetString());
         batchWidth = (size_t)strWidth.Atoi();
      } break;
      }
      ++idxToken;
   }

   this->SetBatchDepth(batchDepth);
   this->SetBatchHeight(batchHeight);
   this->SetBatchWidth(batchWidth);
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
      case 1: // depth
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
   int depth = 0;
   int height = 0;
   int width = 0;
   bool flattening = false;

   // Split layer details
   TObjArray *subStrings = layerString.Tokenize(delim);
   TIter nextToken(subStrings);
   TObjString *token = (TObjString *)nextToken();
   int idxToken = 0;

   for (; token != nullptr; token = (TObjString *)nextToken()) {
      switch (idxToken) {
      case 1: {
         TString strDepth(token->GetString());
         depth = strDepth.Atoi();
      } break;
      case 2: // height
      {
         TString strHeight(token->GetString());
         height = strHeight.Atoi();
      } break;
      case 3: // width
      {
         TString strWidth(token->GetString());
         width = strWidth.Atoi();
      } break;
      case 4: // flattening
      {
         TString flat(token->GetString());
         if (flat == "FLAT") {
            flattening = true;
         }
      } break;
      }
      ++idxToken;
   }

   // Add the reshape layer
   TReshapeLayer<Architecture_t> *reshapeLayer = deepNet.AddReshapeLayer(depth, height, width, flattening);
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
   : MethodBase(jobName, Types::kDL, methodTitle, theData, theOption), fInputDepth(), fInputHeight(), fInputWidth(),
     fBatchDepth(), fBatchHeight(), fBatchWidth(), fWeightInitialization(), fOutputFunction(), fLossFunction(),
     fInputLayoutString(), fBatchLayoutString(), fLayoutString(), fErrorStrategy(), fTrainingStrategyString(),
     fWeightInitializationString(), fArchitectureString(), fResume(false), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a weight file.
MethodDL::MethodDL(DataSetInfo &theData, const TString &theWeightFile)
   : MethodBase(Types::kDL, theData, theWeightFile), fInputDepth(), fInputHeight(), fInputWidth(), fBatchDepth(),
     fBatchHeight(), fBatchWidth(), fWeightInitialization(), fOutputFunction(), fLossFunction(), fInputLayoutString(),
     fBatchLayoutString(), fLayoutString(), fErrorStrategy(), fTrainingStrategyString(), fWeightInitializationString(),
     fArchitectureString(), fResume(false), fTrainingSettings()
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
   KeyValueVector_t blockKeyValues;
   const TString keyValueDelim("=");

   TObjArray *blockStrings = parseString.Tokenize(blockDelim);
   TIter nextBlock(blockStrings);
   TObjString *blockString = (TObjString *)nextBlock();

   for (; blockString != nullptr; blockString = (TObjString *)nextBlock()) {
      blockKeyValues.push_back(std::map<TString, TString>());
      std::map<TString, TString> &currentBlock = blockKeyValues.back();

      TObjArray *subStrings = blockString->GetString().Tokenize(tokenDelim);
      TIter nextToken(subStrings);
      TObjString *token = (TObjString *)nextToken();

      for (; token != nullptr; token = (TObjString *)nextToken()) {
         TString strKeyValue(token->GetString());
         int delimPos = strKeyValue.First(keyValueDelim.Data());
         if (delimPos <= 0) continue;

         TString strKey = TString(strKeyValue(0, delimPos));
         strKey.ToUpper();
         TString strValue = TString(strKeyValue(delimPos + 1, strKeyValue.Length()));

         strKey.Strip(TString::kBoth, ' ');
         strValue.Strip(TString::kBoth, ' ');

         currentBlock.insert(std::make_pair(strKey, strValue));
      }
   }
   return blockKeyValues;
}

////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the CNN
Bool_t MethodDL::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass) return kTRUE;
   if (type == Types::kRegression) return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::Train()
{
   if (fInteractive) {
      Log() << kFATAL << "Not implemented yet" << Endl;
      return;
   }

   if (this->GetArchitectureString() == "GPU") {
      TrainGpu();
      return;
   } else if (this->GetArchitectureString() == "OpenCL") {
      Log() << kFATAL << "OpenCL backend not yet supported." << Endl;
      return;
   } else if (this->GetArchitectureString() == "CPU") {
      TrainCpu();
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::TrainGpu()
{
#ifdef DNNCUDA // Included only if DNNCUDA flag is set.

   using Architecture_t = DNN::TCuda<Double_t>;
   using Scalar_t = Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
   using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

   Log() << kINFO << "Start of deep neural network training on GPU." << Endl << Endl;

   // Determine the number of training and testing examples
   size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
   size_t nTestSamples = GetEventCollection(Types::kTesting).size();

   // Determine the number of outputs
   size_t outputSize = 1;
   if (fAnalysisType == Types::kRegression && GetNTargets() != 0) {
      outputSize = GetNTargets();
   } else if (fAnalysisType == Types::kMulticlass && DataInfo().GetNClasses() >= 2) {
      outputSize = DataInfo().GetNClasses();
   }

   size_t trainingPhase = 1;
   for (TTrainingSettings &settings : this->GetTrainingSettings()) {

      size_t nThreads = 1;

      Log() << "Training phase " << trainingPhase << " of " << this->GetTrainingSettings().size() << ":" << Endl;
      trainingPhase++;

      // After the processing of the options, initialize the master deep net
      size_t batchSize = settings.batchSize;
      // Should be replaced by actual implementation. No support for this now.
      size_t inputDepth = this->GetInputDepth();
      size_t inputHeight = this->GetInputHeight();
      size_t inputWidth = this->GetInputWidth();
      size_t batchDepth = this->GetBatchDepth();
      size_t batchHeight = this->GetBatchHeight();
      size_t batchWidth = this->GetBatchWidth();
      ELossFunction J = this->GetLossFunction();
      EInitialization I = this->GetWeightInitialization();
      ERegularization R = settings.regularization;
      Scalar_t weightDecay = settings.weightDecay;
      DeepNet_t deepNet(batchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R,
                        weightDecay);

      // Initialize the vector of slave nets
      std::vector<DeepNet_t> nets{};
      nets.reserve(nThreads);
      for (size_t i = 0; i < nThreads; i++) {
         // create a copies of the master deep net
         nets.push_back(deepNet);
      }

      // Add all appropriate layers
      CreateDeepNet(deepNet, nets);

      // Loading the training and testing datasets
      TensorDataLoader_t trainingData(GetEventCollection(Types::kTraining), nTrainingSamples, deepNet.GetBatchSize(),
                                      deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                      deepNet.GetOutputWidth(), nThreads);

      TensorDataLoader_t testingData(GetEventCollection(Types::kTesting), nTestSamples, deepNet.GetBatchSize(),
                                     deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                     deepNet.GetOutputWidth(), nThreads);

      // Initialize the minimizer
      DNN::TDLGradientDescent<Architecture_t> minimizer(settings.learningRate, settings.convergenceSteps,
                                                        settings.testInterval);

      // Initialize the vector of batches, one batch for one slave network
      std::vector<TTensorBatch<Architecture_t>> batches{};

      bool converged = false;
      // count the steps until the convergence
      size_t stepCount = 0;
      size_t batchesInEpoch = nTrainingSamples / deepNet.GetBatchSize();

      // start measuring
      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      if (!fInteractive) {
         Log() << std::setw(10) << "Epoch"
               << " | " << std::setw(12) << "Train Err." << std::setw(12) << "Test  Err." << std::setw(12) << "GFLOP/s"
               << std::setw(12) << "Conv. Steps" << Endl;
         std::string separator(62, '-');
         Log() << separator << Endl;
      }

      while (!converged) {
         stepCount++;
         trainingData.Shuffle();

         // execute all epochs
         for (size_t i = 0; i < batchesInEpoch; i += nThreads) {
            // Clean and load new batches, one batch for one slave net
            batches.clear();
            batches.reserve(nThreads);
            for (size_t j = 0; j < nThreads; j++) {
               batches.push_back(trainingData.GetTensorBatch());
            }

            // execute one minimization step
            if (settings.momentum > 0.0) {
               minimizer.StepMomentum(deepNet, nets, batches, settings.momentum);
            } else {
               minimizer.Step(deepNet, nets, batches);
            }
         }

         if ((stepCount % minimizer.GetTestInterval()) == 0) {
            // Compute test error.
            Double_t testError = 0.0;
            for (auto batch : testingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               testError += deepNet.Loss(inputTensor, outputMatrix);
            }

            testError /= (Double_t)(nTestSamples / settings.batchSize);

            // stop measuring
            end = std::chrono::system_clock::now();

            // Compute training error.
            Double_t trainingError = 0.0;
            for (auto batch : trainingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               trainingError += deepNet.Loss(inputTensor, outputMatrix);
            }
            trainingError /= (Double_t)(nTrainingSamples / settings.batchSize);

            // Compute numerical throughput.
            std::chrono::duration<double> elapsed_seconds = end - start;
            double seconds = elapsed_seconds.count();
            double nFlops = (double)(settings.testInterval * batchesInEpoch);
            // nFlops *= deepNet.GetNFlops() * 1e-9;

            converged = minimizer.HasConverged(testError);

            Log() << std::setw(10) << stepCount << " | " << std::setw(12) << trainingError << std::setw(12) << testError
                  << std::setw(12) << nFlops / seconds << std::setw(12) << minimizer.GetConvergenceCount() << Endl;

            if (converged) {
               Log() << Endl;
            }
         }
      }
   }

#else  // DNNCUDA flag not set.

   Log() << kFATAL << "CUDA backend not enabled. Please make sure "
                      "you have CUDA installed and it was successfully "
                      "detected by CMAKE."
         << Endl;
#endif // DNNCUDA
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::TrainCpu()
{
#ifdef DNNCPU // Included only if DNNCPU flag is set.

   std::cout << "Train CPU Method" << std::endl;
   using Architecture_t = DNN::TCpu<Double_t>;
   using Scalar_t = Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t>;
   using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

   Log() << kINFO << "Start of deep neural network training on CPU." << Endl << Endl;

   // Determine the number of training and testing examples
   size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
   size_t nTestSamples = GetEventCollection(Types::kTesting).size();

   // Determine the number of outputs
   size_t outputSize = 1;
   if (fAnalysisType == Types::kRegression && GetNTargets() != 0) {
      outputSize = GetNTargets();
   } else if (fAnalysisType == Types::kMulticlass && DataInfo().GetNClasses() >= 2) {
      outputSize = DataInfo().GetNClasses();
   }

   size_t trainingPhase = 1;
   for (TTrainingSettings &settings : this->GetTrainingSettings()) {

      size_t nThreads = 1;

      Log() << "Training phase " << trainingPhase << " of " << this->GetTrainingSettings().size() << ":" << Endl;
      trainingPhase++;

      // After the processing of the options, initialize the master deep net
      size_t batchSize = settings.batchSize;
      // Should be replaced by actual implementation. No support for this now.
      size_t inputDepth = this->GetInputDepth();
      size_t inputHeight = this->GetInputHeight();
      size_t inputWidth = this->GetInputWidth();
      size_t batchDepth = this->GetBatchDepth();
      size_t batchHeight = this->GetBatchHeight();
      size_t batchWidth = this->GetBatchWidth();
      ELossFunction J = this->GetLossFunction();
      EInitialization I = this->GetWeightInitialization();
      ERegularization R = settings.regularization;
      Scalar_t weightDecay = settings.weightDecay;
      DeepNet_t deepNet(batchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R,
                        weightDecay);

      // Initialize the vector of slave nets
      std::vector<DeepNet_t> nets{};
      nets.reserve(nThreads);
      for (size_t i = 0; i < nThreads; i++) {
         // create a copies of the master deep net
         nets.push_back(deepNet);
      }

      // Add all appropriate layers
      CreateDeepNet(deepNet, nets);

      // Loading the training and testing datasets
      TensorDataLoader_t trainingData(GetEventCollection(Types::kTraining), nTrainingSamples, deepNet.GetBatchSize(),
                                      deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                      deepNet.GetOutputWidth(), nThreads);

      TensorDataLoader_t testingData(GetEventCollection(Types::kTesting), nTestSamples, deepNet.GetBatchSize(),
                                     deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth(),
                                     deepNet.GetOutputWidth(), nThreads);

      // Initialize the minimizer
      DNN::TDLGradientDescent<Architecture_t> minimizer(settings.learningRate, settings.convergenceSteps,
                                                        settings.testInterval);

      // Initialize the vector of batches, one batch for one slave network
      std::vector<TTensorBatch<Architecture_t>> batches{};

      bool converged = false;
      // count the steps until the convergence
      size_t stepCount = 0;
      size_t batchesInEpoch = nTrainingSamples / deepNet.GetBatchSize();

      // start measuring
      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();

      if (!fInteractive) {
         Log() << std::setw(10) << "Epoch"
               << " | " << std::setw(12) << "Train Err." << std::setw(12) << "Test  Err." << std::setw(12) << "GFLOP/s"
               << std::setw(12) << "Conv. Steps" << Endl;
         std::string separator(62, '-');
         Log() << separator << Endl;
      }

      while (!converged) {
         stepCount++;
         trainingData.Shuffle();

         // execute all epochs
         for (size_t i = 0; i < batchesInEpoch; i += nThreads) {
            // Clean and load new batches, one batch for one slave net
            batches.clear();
            batches.reserve(nThreads);
            for (size_t j = 0; j < nThreads; j++) {
               batches.push_back(trainingData.GetTensorBatch());
            }

            // execute one minimization step
            if (settings.momentum > 0.0) {
               minimizer.StepMomentum(deepNet, nets, batches, settings.momentum);
            } else {
               minimizer.Step(deepNet, nets, batches);
            }
         }

         if ((stepCount % minimizer.GetTestInterval()) == 0) {
            // Compute test error.
            Double_t testError = 0.0;
            for (auto batch : testingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();
               testError += deepNet.Loss(inputTensor, outputMatrix, weights);
            }

            testError /= (Double_t)(nTestSamples / settings.batchSize);

            // stop measuring
            end = std::chrono::system_clock::now();

            // Compute training error.
            Double_t trainingError = 0.0;
            for (auto batch : trainingData) {
               auto inputTensor = batch.GetInput();
               auto outputMatrix = batch.GetOutput();
               auto weights = batch.GetWeights();
               trainingError += deepNet.Loss(inputTensor, outputMatrix, weights);
            }
            trainingError /= (Double_t)(nTrainingSamples / settings.batchSize);

            // Compute numerical throughput.
            std::chrono::duration<double> elapsed_seconds = end - start;
            double seconds = elapsed_seconds.count();
            double nFlops = (double)(settings.testInterval * batchesInEpoch);
            // nFlops *= net.GetNFlops() * 1e-9;

            converged = minimizer.HasConverged(testError);

            Log() << std::setw(10) << stepCount << " | " << std::setw(12) << trainingError << std::setw(12) << testError
                  << std::setw(12) << nFlops / seconds << std::setw(12) << minimizer.GetConvergenceCount() << Endl;

            if (converged) {
               Log() << Endl;
            }
         }
      }
   }

#else  // DNNCPU flag not set.
   Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                      "you have a BLAS implementation and it was successfully "
                      "detected by CMake as well that the imt CMake flag is set."
         << Endl;
#endif // DNNCPU
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