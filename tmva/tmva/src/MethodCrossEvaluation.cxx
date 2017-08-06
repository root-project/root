// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard v. Toerne, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCrossEvaluation                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Doug Schouten   <dschoute@sfu.ca>        - Simon Fraser U., Canada        *
 *      Jan Therhaag    <jan.therhaag@cern.ch>   - U. of Bonn, Germany            *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::MethodCrossEvaluation
\ingroup TMVA
*/
#include "TMVA/MethodCrossEvaluation.h"

#include "TMVA/ClassifierFactory.h"

REGISTER_METHOD(CrossEvaluation)

ClassImp(TMVA::MethodCrossEvaluation);

const Int_t TMVA::MethodCrossEvaluation::fgDebugLevel = 0;

////////////////////////////////////////////////////////////////////////////////
/// The standard constructor for the "boosted decision trees".

TMVA::MethodCrossEvaluation::MethodCrossEvaluation( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption ) :
   TMVA::MethodBase( jobName, Types::kCrossEvaluation, methodTitle, theData, theOption)
{
}

////////////////////////////////////////////////////////////////////////////////

TMVA::MethodCrossEvaluation::MethodCrossEvaluation( DataSetInfo& theData,
                            const TString& theWeightFile)
   : TMVA::MethodBase( Types::kCrossEvaluation, theData, theWeightFile)
{
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::MethodCrossEvaluation::DeclareOptions()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Options that are used ONLY for the READER to ensure backward compatibility.

void TMVA::MethodCrossEvaluation::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
}

////////////////////////////////////////////////////////////////////////////////
/// The option string is decoded, for available options see "DeclareOptions".

void TMVA::MethodCrossEvaluation::ProcessOptions()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Common initialisation with defaults for the BDT-Method.

void TMVA::MethodCrossEvaluation::Init( void )
{
}


////////////////////////////////////////////////////////////////////////////////
/// Reset the method, as if it had just been instantiated (forget all training etc.).

void TMVA::MethodCrossEvaluation::Reset( void )
{
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor.
///
///  - Note: fEventSample and ValidationSample are already deleted at the end of TRAIN
///         When they are not used anymore

TMVA::MethodCrossEvaluation::~MethodCrossEvaluation( void )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Call the Optimizer with the set of parameters and ranges that
/// are meant to be tuned.

std::map<TString,Double_t>  TMVA::MethodCrossEvaluation::OptimizeTuningParameters(TString fomType, TString fitType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set the tuning parameters according to the argument.

void TMVA::MethodCrossEvaluation::SetTuneParameters(std::map<TString,Double_t> tuneParameters)
{
}

////////////////////////////////////////////////////////////////////////////////
///  training.

void TMVA::MethodCrossEvaluation::Train()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Write weights to XML.

void TMVA::MethodCrossEvaluation::AddWeightsXMLTo( void* parent ) const
{
}

////////////////////////////////////////////////////////////////////////////////
/// Reads from the xml file.
/// 

void TMVA::MethodCrossEvaluation::ReadWeightsFromXML(void* parent)
{

}

////////////////////////////////////////////////////////////////////////////////
/// Read the weights
/// 

void  TMVA::MethodCrossEvaluation::ReadWeightsFromStream( std::istream& istr )
{

}

////////////////////////////////////////////////////////////////////////////////
///

Double_t TMVA::MethodCrossEvaluation::GetMvaValue( Double_t* err, Double_t* errUpper ){
}

////////////////////////////////////////////////////////////////////////////////
/// Get the multiclass MVA response.

const std::vector<Float_t>& TMVA::MethodCrossEvaluation::GetMulticlassValues()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Get the regression value generated by the containing methods.

const std::vector<Float_t> & TMVA::MethodCrossEvaluation::GetRegressionValues()
{

}

////////////////////////////////////////////////////////////////////////////////
/// Here we could write some histograms created during the processing
/// to the output file.

void  TMVA::MethodCrossEvaluation::WriteMonitoringHistosToFile( void ) const
{
   // Used for evaluation, which is outside the life time of MethodCrossEval.
   Log() << kFATAL << "Method CrossEvaluation should not be created manually,"
                      " only as part of using TMVA::Reader." << Endl;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Get help message text.

void TMVA::MethodCrossEvaluation::GetHelpMessage() const
{
   // Log() << Endl;
   // Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << "By the nature of the binary splits performed on the individual" << Endl;
   // Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   // Log() << Endl;
   // Log() << "The two most important parameters in the configuration are the  " << Endl;

}

////////////////////////////////////////////////////////////////////////////////

const TMVA::Ranking * TMVA::MethodCrossEvaluation::CreateRanking()
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TMVA::MethodCrossEvaluation::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Make ROOT-independent C++ class for classifier response (classifier-specific implementation).

void TMVA::MethodCrossEvaluation::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{

}

////////////////////////////////////////////////////////////////////////////////
/// Specific class header.

void TMVA::MethodCrossEvaluation::MakeClassSpecificHeader(  std::ostream& fout, const TString& className) const
{

}

