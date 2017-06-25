// @(#)root/tmva $Id$
// Author: Vladimir Ilievski 20/06/2017


/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/MethodDL.h"
#include "TMVA/Types.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/MethodBase.h"

#include <iostream>

ClassImp(TMVA::MethodDL);

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA
{

////////////////////////////////////////////////////////////////////////////////
TString fetchValueTmp (const std::map<TString, TString>& keyValueMap, TString key)
{
   key.ToUpper ();
   std::map<TString, TString>::const_iterator it = keyValueMap.find (key);
   if (it == keyValueMap.end()) {
      return TString ("");
   }
   return it->second;
}
    
////////////////////////////////////////////////////////////////////////////////
template <typename T>
T fetchValueTmp(const std::map<TString,TString>& keyValueMap,
                TString key,
                T defaultValue);

////////////////////////////////////////////////////////////////////////////////
template <>
int fetchValueTmp(const std::map<TString,TString>& keyValueMap,
                  TString key,
                  int defaultValue)
{
   TString value (fetchValueTmp (keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atoi ();
}
    
////////////////////////////////////////////////////////////////////////////////
template <>
double fetchValueTmp (const std::map<TString,TString>& keyValueMap,
                      TString key, double defaultValue)
{
   TString value (fetchValueTmp (keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value.Atof ();
}
    
////////////////////////////////////////////////////////////////////////////////
template <>
TString fetchValueTmp (const std::map<TString,TString>& keyValueMap,
                       TString key, TString defaultValue)
{
   TString value (fetchValueTmp (keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
   return value;
}
    
////////////////////////////////////////////////////////////////////////////////
template <>
bool fetchValueTmp (const std::map<TString,TString>& keyValueMap,
                    TString key, bool defaultValue)
{
   TString value (fetchValueTmp (keyValueMap, key));
   if (value == "") {
      return defaultValue;
   }
        
   value.ToUpper ();
   if (value == "TRUE" || value == "T" || value == "1") {
      return true;
   }
        
   return false;
}
    
////////////////////////////////////////////////////////////////////////////////
template <>
std::vector<double> fetchValueTmp(const std::map<TString, TString> & keyValueMap,
                                  TString key,
                                  std::vector<double> defaultValue)
{
   TString parseString (fetchValueTmp (keyValueMap, key));
   if (parseString == "") {
      return defaultValue;
   }
        
   parseString.ToUpper ();
   std::vector<double> values;
        
   const TString tokenDelim ("+");
   TObjArray* tokenStrings = parseString.Tokenize (tokenDelim);
   TIter nextToken (tokenStrings);
   TObjString* tokenString = (TObjString*)nextToken ();
   for (; tokenString != NULL; tokenString = (TObjString*)nextToken ()) {
      std::stringstream sstr;
      double currentValue;
      sstr << tokenString->GetString ().Data ();
      sstr >> currentValue;
      values.push_back (currentValue);
   }
   return values;
}
    
    
////////////////////////////////////////////////////////////////////////////////
/// standard constructor
MethodDL::MethodDL(const TString& jobName,
                   Types::EMVA mvaType,
                   const TString&  methodTitle,
                   DataSetInfo& theData,
                   const TString& theOption)
   : MethodBase(jobName, mvaType, methodTitle, theData, theOption),
     fWeightInitialization(), fOutputFunction(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
MethodDL::MethodDL(Types::EMVA mvaType,
                   DataSetInfo& theData,
                   const TString& theWeightFile)
    : MethodBase(mvaType, theData, theWeightFile),
      fWeightInitialization(), fOutputFunction(), fErrorStrategy(),
      fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
      fResume(false), fTrainingSettings()
{
   // Nothing to do here
}
   
////////////////////////////////////////////////////////////////////////////////
/// destructor
MethodDL::~MethodDL()
{
   // Nothing to do here
}

void MethodDL::DeclareOptions()
{
    DeclareOptionRef(fErrorStrategy="CROSSENTROPY",
                     "ErrorStrategy",
                     "Loss function: Mean squared error (regression)"
                     " or cross entropy (binary classification).");
    AddPreDefVal(TString("CROSSENTROPY"));
    AddPreDefVal(TString("SUMOFSQUARES"));
    AddPreDefVal(TString("MUTUALEXCLUSIVE"));
    
    DeclareOptionRef(fWeightInitializationString="XAVIER",
                     "WeightInitialization",
                     "Weight initialization strategy");
    AddPreDefVal(TString("XAVIER"));
    AddPreDefVal(TString("XAVIERUNIFORM"));
    
    DeclareOptionRef(fArchitectureString = "CPU", "Architecture", "Which architecture to perform the training on.");
    AddPreDefVal(TString("STANDARD"));
    AddPreDefVal(TString("CPU"));
    AddPreDefVal(TString("GPU"));
    AddPreDefVal(TString("OPENCL"));
    
   DeclareOptionRef(
      fTrainingStrategyString = "LearningRate=1e-1,"
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
                                "TrainingStrategy",
                                "Defines the training strategies.");


}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::CallDeclareOptions()
{
   this -> DeclareOptions();
}
    
////////////////////////////////////////////////////////////////////////////////
void MethodDL::ProcessOptions()
{
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO
      << "Will ignore negative events in training!"
      << Endl;
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

   // Loss function and output.
   fOutputFunction = EOutputFunction::kSigmoid;
   if (fAnalysisType == Types::kClassification){
      if (fErrorStrategy == "SUMOFSQUARES") {
         fLossFunction = ELossFunction::kMeanSquaredError;
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fLossFunction =  ELossFunction::kCrossEntropy;
      }
      fOutputFunction = EOutputFunction::kSigmoid;
   } else if (fAnalysisType == Types::kRegression) {
      if (fErrorStrategy != "SUMOFSQUARES") {
         Log () << kWARNING << "For regression only SUMOFSQUARES is a valid "
                << " neural net error function. Setting error function to "
                << " SUMOFSQUARES now." << Endl;
      }
        
      fLossFunction =  ELossFunction::kMeanSquaredError;
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
   }
   else if (fWeightInitializationString == "XAVIERUNIFORM") {
      fWeightInitialization = DNN::EInitialization::kUniform;
   }
   else {
      fWeightInitialization = DNN::EInitialization::kGauss;
   }
    
   // Training settings.
    
   KeyValueVector_t strategyKeyValues = ParseKeyValueString(fTrainingStrategyString,
                                                            TString ("|"),
                                                            TString (","));
   for (auto& block : strategyKeyValues) {
      TTrainingSettings settings;
        
      settings.convergenceSteps     = fetchValueTmp(block, "ConvergenceSteps", 100);
      settings.batchSize            = fetchValueTmp(block, "BatchSize", 30);
      settings.testInterval         = fetchValueTmp(block, "TestRepetitions", 7);
      settings.weightDecay          = fetchValueTmp(block, "WeightDecay", 0.0);
      settings.learningRate         = fetchValueTmp(block, "LearningRate", 1e-5);
      settings.momentum             = fetchValueTmp(block, "Momentum", 0.3);
      settings.dropoutProbabilities = fetchValueTmp(block, "DropConfig",
                                                 std::vector<Double_t>());
        
      TString regularization = fetchValueTmp(block, "Regularization",
                                          TString ("NONE"));
      if (regularization == "L1") {
         settings.regularization = DNN::ERegularization::kL1;
      } else if (regularization == "L2") {
         settings.regularization = DNN::ERegularization::kL2;
      }
        
      TString strMultithreading = fetchValueTmp(block, "Multithreading",
                                             TString ("True"));
       
      if (strMultithreading.BeginsWith ("T")) {
         settings.multithreading = true;
      } else {
         settings.multithreading = false;
      }
        
      fTrainingSettings.push_back(settings);
   }
}

////////////////////////////////////////////////////////////////////////////////
void MethodDL::CallProcessOptions()
{
    this -> ProcessOptions();
}
    

////////////////////////////////////////////////////////////////////////////////
/// parse key value pairs in blocks -> return vector of blocks with map of key value pairs
auto MethodDL::ParseKeyValueString(TString parseString,
                                   TString blockDelim,
                                   TString tokenDelim)
-> KeyValueVector_t
{
   KeyValueVector_t blockKeyValues;
   const TString keyValueDelim ("=");
        
   TObjArray* blockStrings = parseString.Tokenize (blockDelim);
   TIter nextBlock (blockStrings);
   TObjString* blockString = (TObjString *) nextBlock();
        
   for(; blockString != nullptr; blockString = (TObjString *) nextBlock()) {
      blockKeyValues.push_back (std::map<TString,TString>());
      std::map<TString,TString>& currentBlock = blockKeyValues.back ();
            
      TObjArray* subStrings = blockString->GetString ().Tokenize (tokenDelim);
      TIter nextToken (subStrings);
      TObjString* token = (TObjString*)nextToken ();
            
      for (; token != nullptr; token = (TObjString *)nextToken()) {
         TString strKeyValue (token->GetString ());
         int delimPos = strKeyValue.First (keyValueDelim.Data ());
         if (delimPos <= 0)
            continue;
                
         TString strKey = TString (strKeyValue (0, delimPos));
         strKey.ToUpper();
         TString strValue = TString (strKeyValue (delimPos+1, strKeyValue.Length ()));
                
         strKey.Strip (TString::kBoth, ' ');
         strValue.Strip (TString::kBoth, ' ');
                
         currentBlock.insert (std::make_pair (strKey, strValue));
      }
   }
   return blockKeyValues;
}

    
} // namespace TMVA
