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
TMVA::MethodCNN::MethodCNN(const TString& jobName,
                           const TString&  methodTitle,
                           DataSetInfo& theData,
                           const TString& theOption)
   : MethodBase( jobName, Types::kCNN, methodTitle, theData, theOption),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
    
{
   // Nothing to do here
}


////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
TMVA::MethodCNN::MethodCNN(DataSetInfo& theData,
                           const TString& theWeightFile)
   : MethodBase( Types::kCNN, theData, theWeightFile),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// destructor
    
TMVA::MethodCNN::~MethodCNN()
{
   // Nothing to do here
}

    
////////////////////////////////////////////////////////////////////////////////
/// default initializations
void TMVA::MethodCNN::Init()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// Options to be set in the option string
void TMVA::MethodCNN::DeclareOptions()
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
/// Parsing and processing all optiions defining the CNN
void TMVA::MethodCNN::ProcessOptions()
{
   // TO DO
}
    
////////////////////////////////////////////////////////////////////////////////
/// parse layout specification string and return a vector, each entry
/// containing the dimension and the activation function of each successive layer
auto TMVA::MethodCNN::ParseLayoutString(TString layoutString)
-> LayoutVector_t
{
    // TO DO
    LayoutVector_t layout;
    return layout;
}


////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the CNN
Bool_t TMVA::MethodCNN::HasAnalysisType(Types::EAnalysisType type,
                                        UInt_t numberClasses,
                                        UInt_t /*numberTargets*/ )
{
   // TO DO
   return true;
}


    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::Train()
{
    
   if (fArchitectureString == "GPU") {
      TrainGpu();
      return;
   } else if (fArchitectureString == "OpenCL") {
      Log() << kFATAL << "OpenCL backend not yet supported." << Endl;
      return;
   } else if (fArchitectureString == "CPU") {
      TrainCpu();
      return;
   }
    
}
    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::TrainGpu()
{
   // TO DO
}
    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::TrainCpu()
{
   // TO DO
}
    
    
////////////////////////////////////////////////////////////////////////////////
Double_t TMVA::MethodCNN::GetMvaValue(Double_t* /*errLower*/,
                                      Double_t* /*errUpper*/ )
{
   // TO DO
   return 0.0;
}
    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::AddWeightsXMLTo( void* parent ) const
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::ReadWeightsFromXML(void* rootXML)
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::ReadWeightsFromStream( std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const TMVA::Ranking* TMVA::MethodCNN::CreateRanking()
{
   // TO DO
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodCNN::GetHelpMessage() const
{
   // TO DO
}
    
} // namespace TMVA
