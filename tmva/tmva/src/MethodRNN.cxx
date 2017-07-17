// @(#)root/tmva/:$Id$
// Author: Saurav Shekhar 21/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRNN                                                             *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Saurav Shekhar    <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 * All rights reserved.                                                           *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

/*! \class TMVA::MethodRNN
\ingroup TMVA
Recurrent Neural Network Implementation.
*/

#include "TMVA/MethodRNN.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Architectures/Reference.h"

#include <iostream>

REGISTER_METHOD(RNN)
ClassImp(TMVA::MethodRNN);

namespace TMVA
{

////////////////////////////////////////////////////////////////////////////////
/// standard constructor
TMVA::MethodRNN::MethodRNN(const TString& jobName,
                           const TString&  methodTitle,
                           DataSetInfo& theData,
                           const TString& theOption)
   : MethodBase( jobName, Types::kRNN, methodTitle, theData, theOption),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
    
{
   // Nothing to do here
}


////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
TMVA::MethodRNN::MethodRNN(DataSetInfo& theData,
                           const TString& theWeightFile)
   : MethodBase( Types::kRNN, theData, theWeightFile),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// destructor
    
TMVA::MethodRNN::~MethodRNN()
{
   // Nothing to do here
}

    
////////////////////////////////////////////////////////////////////////////////
/// default initializations
void TMVA::MethodRNN::Init()
{
   // Nothing to do here
}
    
////////////////////////////////////////////////////////////////////////////////
/// Options to be set in the option string
void TMVA::MethodRNN::DeclareOptions()
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
/// Parsing and processing all optiions defining the RNN
void TMVA::MethodRNN::ProcessOptions()
{
   // TO DO
}
    
////////////////////////////////////////////////////////////////////////////////
/// parse layout specification string and return a vector, each entry
/// containing the dimension and the activation function of each successive layer
auto TMVA::MethodRNN::ParseLayoutString(TString layoutString)
-> LayoutVector_t
{
    // TO DO
    LayoutVector_t layout;
    return layout;
}


////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the RNN
Bool_t TMVA::MethodRNN::HasAnalysisType(Types::EAnalysisType type,
                                        UInt_t numberClasses,
                                        UInt_t /*numberTargets*/ )
{
   // TO DO
   return true;
}


    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::Train()
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
void TMVA::MethodRNN::TrainGpu()
{
   // TO DO
}
    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::TrainCpu()
{
   // TO DO
}
    
    
////////////////////////////////////////////////////////////////////////////////
Double_t TMVA::MethodRNN::GetMvaValue(Double_t* /*errLower*/,
                                      Double_t* /*errUpper*/ )
{
   // TO DO
   return 0.0;
}
    
////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::AddWeightsXMLTo( void* parent ) const
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::ReadWeightsFromXML(void* rootXML)
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::ReadWeightsFromStream( std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const TMVA::Ranking* TMVA::MethodRNN::CreateRanking()
{
   // TO DO
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodRNN::GetHelpMessage() const
{
   // TO DO
}
    
} // namespace TMVA
