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

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodDL.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/DLMinimizers.h"

REGISTER_METHOD(DL)
ClassImp(TMVA::MethodDL);

namespace TMVA {

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