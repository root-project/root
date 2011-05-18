// @(#)root/tmva $Id$    
// Author: Marcin .... 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBayesClassifier                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Abhishek Narain, <narainabhi@gmail.com> - University of Houston           *
 *                                                                                *
 * Copyright (c) 2005-2006:                                                       *
 *      University of Houston,                                                    *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// ... description of bayesian classifiers ...
//_______________________________________________________________________

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodBayesClassifier.h"
#include "TMVA/Tools.h"
#include "Riostream.h"

REGISTER_METHOD(BayesClassifier)

ClassImp(TMVA::MethodBayesClassifier)

//_______________________________________________________________________
TMVA::MethodBayesClassifier::MethodBayesClassifier( const TString& jobName,
                                                    const TString& methodTitle,
                                                    DataSetInfo& theData, 
                                                    const TString& theOption,
                                                    TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kBayesClassifier, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodBayesClassifier::MethodBayesClassifier( DataSetInfo& theData, 
                                                    const TString& theWeightFile,  
                                                    TDirectory* theTargetDir ) :
   TMVA::MethodBase( Types::kBayesClassifier, theData, theWeightFile, theTargetDir ) 
{
   // constructor from weight file
}

//_______________________________________________________________________
Bool_t TMVA::MethodBayesClassifier::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // Variable can handle classification with 2 classes 
   if( type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodBayesClassifier::Init( void )
{
   // default initialisation
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::ProcessOptions() 
{
   // the option string is decoded, for availabel options see "DeclareOptions"
}

//_______________________________________________________________________
TMVA::MethodBayesClassifier::~MethodBayesClassifier( void )
{
   // destructor
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::Train( void )
{
   // some training 
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::AddWeightsXMLTo( void* /*parent*/ ) const {
   Log() << kFATAL << "Please implement writing of weights as XML" << Endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodBayesClassifier::ReadWeightsFromStream( istream & )
{
   // read back the training results from a file (stream)
}

//_______________________________________________________________________
Double_t TMVA::MethodBayesClassifier::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // returns MVA value for given event
   Double_t myMVA = 0;

   // cannot determine error
   NoErrorCalc(err, errUpper);

   return myMVA;
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << endl;
   fout << "};" << endl;
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "<None>" << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "<None>" << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "<None>" << Endl;
}
