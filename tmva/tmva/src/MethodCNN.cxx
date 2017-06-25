// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 07/06/2017

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/MethodCNN.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Architectures/Reference.h"

#include <iostream>

REGISTER_METHOD(CNN)
ClassImp(TMVA::MethodCNN)


using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;


namespace TMVA
{

////////////////////////////////////////////////////////////////////////////////
/// standard constructor
MethodCNN::MethodCNN(const TString& jobName,
                     const TString&  methodTitle,
                     DataSetInfo& theData,
                     const TString& theOption)
   : MethodDL(jobName, Types::kCNN, methodTitle, theData, theOption),
    fLayoutString()
{
   // Nothing to do here
}


////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
MethodCNN::MethodCNN(DataSetInfo& theData,
                     const TString& theWeightFile)
   : MethodDL(Types::kCNN, theData, theWeightFile), fLayoutString()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// destructor
    
MethodCNN::~MethodCNN()
{
   // Nothing to do here
}

    
////////////////////////////////////////////////////////////////////////////////
/// default initializations
void MethodCNN::Init()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// Options to be set in the option string
void MethodCNN::DeclareOptions()
{
   MethodDL::CallDeclareOptions();
}

////////////////////////////////////////////////////////////////////////////////
/// Parsing and processing all optiions defining the CNN
void MethodCNN::ProcessOptions()
{
   MethodDL::CallProcessOptions();
    
   // Set Network structure
   ParseLayoutString(fLayoutString);
    
   // Should be replaced by actual implementation. No support for this now.
   size_t inputDepth = 0;
   size_t inputHeight = 0;
   size_t inputWidth = 0;
    
   size_t outputSize = 1;
   if (fAnalysisType == Types::kRegression && GetNTargets() != 0) {
      outputSize = GetNTargets();
   } else if (fAnalysisType == Types::kMulticlass && DataInfo().GetNClasses() >= 2) {
      outputSize = DataInfo().GetNClasses();
   }
    
   // Because the size of the batch will change
   fConvNet.SetBatchSize(1);
   fConvNet.SetInputDepth(inputDepth);
   fConvNet.SetInputHeight(inputHeight);
   fConvNet.SetInputWidth(inputWidth);
    
   auto itLayout    = std::begin (fLayout);
   auto itLayoutEnd = std::end (fLayout)-1;
   for ( ; itLayout != itLayoutEnd; ++itLayout) {
      ECNNLayerType currLayerType = std::get<0>(*itLayout);
      int currLayerDepth = std::get<1>(*itLayout);
      int currLayerFltHeight = std::get<2>(*itLayout);
      int currLayerFltWidth = std::get<3>(*itLayout);
      int currLayerStrRows = std::get<4>(*itLayout);
      int currLayerStrCols = std::get<5>(*itLayout);
      int currLayerPadHeight = std::get<6>(*itLayout);
      int currLayerPadWidth = std::get<7>(*itLayout);
      EActivationFunction currLayerActFnc = std::get<8>(*itLayout);
       
      switch(currLayerType)
      {
         case kConv:
         {
            fConvNet.AddConvLayer(currLayerDepth, currLayerFltHeight,
                                  currLayerFltWidth, currLayerStrRows,
                                  currLayerStrCols, currLayerPadHeight,
                                  currLayerPadWidth, currLayerActFnc);
         }
         break;
         case kPool:
         {
            fConvNet.AddPoolLayer(currLayerFltHeight, currLayerFltWidth,
                                  currLayerStrRows, currLayerStrCols);
         }
         break;
         case kFC:
         {
            fConvNet.AddFullyConnLayer(currLayerDepth, currLayerActFnc);
         }
         break;
      }
   }
   fConvNet.AddFullyConnLayer(outputSize, EActivationFunction::kIdentity);
    
    
   // Loss function and output.
   fConvNet.SetLossFunction(this -> GetLossFunction());
}
    
////////////////////////////////////////////////////////////////////////////////
/// parse layout specification string and return a vector, each entry
/// containing the dimension and the activation function of each successive layer
auto MethodCNN::ParseLayoutString(TString layoutString)
-> void
{
   // Layer specification, layer details
   const TString layerDelimiter(",");
   const TString subDelimiter("|");
    
   // Split layers
   TObjArray* layerStrings = layoutString.Tokenize(layerDelimiter);
   TIter nextLayer (layerStrings);
   TObjString* layerString = (TObjString*) nextLayer();
    
   for(; layerString != nullptr; layerString = (TObjString*) nextLayer()) {
        
      
      ECNNLayerType layerType = ECNNLayerType::kConv;
      int depth = 0;
      int fltHeight = 0;
      int fltWidth = 0;
      int strideRows = 0;
      int strideCols = 0;
      int zeroPadHeight = 0;
      int zeroPadWidth = 0;
      EActivationFunction activationFunction = EActivationFunction::kTanh;
       
       
      // Split layer details
      TObjArray* subStrings = layerString->GetString().Tokenize(subDelimiter);
      TIter nextToken (subStrings);
      TObjString* token = (TObjString *) nextToken();
      int idxToken = 0;
       
      for (; token != nullptr; token = (TObjString *) nextToken()) {
         switch(idxToken)
         {
            case 0: // layer type
            {
               TString strLayerType = (token -> GetString());
               if(strLayerType == "CONV") {
                  layerType = ECNNLayerType::kConv;
               } else if(strLayerType == "POOL") {
                  layerType = ECNNLayerType::kPool;
               } else if(strLayerType == "FC") {
                  layerType = ECNNLayerType::kFC;
               }
             }
             break;
             case 1: // depth or width
             {
                TString strDepth (token->GetString ());
                 depth = strDepth.Atoi();
             }
             break;
             case 2: // filter height
             {
                TString strFltHeight (token->GetString ());
                fltHeight = strFltHeight.Atoi();
             }
             break;
             case 3: // filter width
             {
                TString strFltWidth (token->GetString ());
                fltWidth = strFltWidth.Atoi();
             }
             break;
             case 4: // stride in rows
             {
                TString strStrideRows (token->GetString ());
                strideRows = strStrideRows.Atoi();
             }
             break;
             case 5: // stride in cols
             {
                TString strStrideCols (token->GetString ());
                strideCols = strStrideCols.Atoi();
             }
             break;
             case 6: // zero padding height
             {
                TString strZeroPadHeight (token->GetString ());
                zeroPadHeight = strZeroPadHeight.Atoi();
             }
             break;
             case 7: // zero padding width
             {
                TString strZeroPadWidth (token->GetString ());
                zeroPadWidth = strZeroPadWidth.Atoi();
             }
             break;
             case 8: // activation function
             {
                TString strActFnc (token->GetString ());
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
             }
            break;
         }
         ++idxToken;
      }
      fLayout.push_back(std::make_tuple(layerType, depth, fltHeight, fltWidth,
                                        strideRows, strideCols,
                                        zeroPadHeight, zeroPadWidth,
                                        activationFunction));
   }
}


////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the CNN
Bool_t MethodCNN::HasAnalysisType(Types::EAnalysisType type,
                                  UInt_t numberClasses,
                                  UInt_t /*numberTargets*/ )
{
   if (type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;
    
   return kFALSE;
}


    
////////////////////////////////////////////////////////////////////////////////
void MethodCNN::Train()
{
   if (fInteractive && fInteractive->NotInitialized()){
      std::vector<TString> titles = {"Error on training set", "Error on test set"};
      fInteractive->Init(titles);
      // JsMVA progress bar maximum (100%)
      fIPyMaxIter = 100;
   }
    
   if (this -> GetArchitectureString() == "GPU") {
      TrainGpu();
      return;
   } else if (this -> GetArchitectureString() == "OpenCL") {
      Log() << kFATAL << "OpenCL backend not yet supported." << Endl;
      return;
   } else if (this -> GetArchitectureString() == "CPU") {
      TrainCpu();
      return;
   }
    
}
    
////////////////////////////////////////////////////////////////////////////////
void MethodCNN::TrainGpu()
{
   // TO DO
}
    
////////////////////////////////////////////////////////////////////////////////
void MethodCNN::TrainCpu()
{
#ifdef DNNCPU // Included only if DNNCPU flag is set.
    
   // We have to find a way
   size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
   size_t nTestSamples     = GetEventCollection(Types::kTesting).size();
    
   Log() << kINFO << "Start of neural network training on CPU." << Endl << Endl;
   
   fConvNet.Initialize(this -> GetWeightInitialization());
   size_t trainingPhase = 1;
    
   for (TTrainingSettings & settings : this -> GetTrainingSettings()) {
      if (fInteractive){
         fInteractive->ClearGraphs();
      }
       
      Log() << "Training phase " << trainingPhase << " of "
      << this -> GetTrainingSettings().size() << ":" << Endl;
      trainingPhase++;
       
       
       
   }
}
    
    
////////////////////////////////////////////////////////////////////////////////
Double_t MethodCNN::GetMvaValue(Double_t* /*errLower*/,
                                Double_t* /*errUpper*/ )
{
   // TO DO
   return 0.0;
}
    
////////////////////////////////////////////////////////////////////////////////
void MethodCNN::AddWeightsXMLTo( void* parent ) const
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void MethodCNN::ReadWeightsFromXML(void* rootXML)
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void MethodCNN::ReadWeightsFromStream( std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const Ranking* TMVA::MethodCNN::CreateRanking()
{
   // TO DO
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void MethodCNN::GetHelpMessage() const
{
   // TO DO
}
    
} // namespace TMVA
