// @(#)root/tmva $Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/MethodDAE.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Architectures/Reference.h"

#include <iostream>

REGISTER_METHOD(DAE)
ClassImp(TMVA::MethodDAE)


using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;


namespace TMVA
{

////////////////////////////////////////////////////////////////////////////////
/// standard constructor
TMVA::MethodDAE::MethodDAE(const TString& jobName,
                           const TString&  methodTitle,
                           DataSetInfo& theData,
                           const TString& theOption)
   : MethodBase( jobName, Types::kDAE, methodTitle, theData, theOption),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()

{
   // Nothing to do here
}


////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
TMVA::MethodDAE::MethodDAE(DataSetInfo& theData,
                           const TString& theWeightFile)
   : MethodBase( Types::kDAE, theData, theWeightFile),
     fWeightInitialization(), fOutputFunction(), fLayoutString(), fErrorStrategy(),
     fTrainingStrategyString(), fWeightInitializationString(), fArchitectureString(),
     fResume(false), fTrainingSettings()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::MethodDAE::~MethodDAE()
{
   // Nothing to do here
}


////////////////////////////////////////////////////////////////////////////////
/// default initializations
void TMVA::MethodDAE::Init()
{
   // Nothing to do here
}

////////////////////////////////////////////////////////////////////////////////
/// Options to be set in the option string
void TMVA::MethodDAE::DeclareOptions()
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
/// Parsing and processing all optiions defining the DAE
void TMVA::MethodDAE::ProcessOptions()
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
/// parse layout specification string and return a vector, each entry
/// containing the dimension and the activation function of each successive layer
auto TMVA::MethodDAE::ParseLayoutString(TString layoutString)
-> LayoutVector_t
{
    // TO DO
    LayoutVector_t layout;
    return layout;
}


////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the DAE
Bool_t TMVA::MethodDAE::HasAnalysisType(Types::EAnalysisType type,
                                        UInt_t numberClasses,
                                        UInt_t /*numberTargets*/ )
{
   // TO DO
   return true;
}



////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::Train()
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
void TMVA::MethodDAE::TrainGpu()
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::TrainCpu()
{
   // TO DO
}


////////////////////////////////////////////////////////////////////////////////
Double_t TMVA::MethodDAE::GetMvaValue(Double_t* /*errLower*/,
                                      Double_t* /*errUpper*/ )
{
   // TO DO
   return 0.0;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::AddWeightsXMLTo( void* parent ) const
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::ReadWeightsFromXML(void* rootXML)
{
   // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::ReadWeightsFromStream( std::istream & /*istr*/)
{
}

////////////////////////////////////////////////////////////////////////////////
const TMVA::Ranking* TMVA::MethodDAE::CreateRanking()
{
   // TO DO
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::GetHelpMessage() const
{
   // TO DO
}

} // namespace TMVA
