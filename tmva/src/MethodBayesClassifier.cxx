// @(#)root/tmva $Id: MethodBayesClassifier.cxx,v 1.5 2007/04/19 06:53:02 brun Exp $    
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

#include "TMVA/MethodBayesClassifier.h"
#include "Riostream.h"

ClassImp(TMVA::MethodBayesClassifier)

//_______________________________________________________________________
TMVA::MethodBayesClassifier::MethodBayesClassifier( TString jobName, TString methodTitle, DataSet& theData, 
                                                    TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   InitBayesClassifier();
}

//_______________________________________________________________________
TMVA::MethodBayesClassifier::MethodBayesClassifier( DataSet& theData, 
                                                    TString theWeightFile,  
                                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor from weight file
   InitBayesClassifier();
}

//_______________________________________________________________________
void TMVA::MethodBayesClassifier::InitBayesClassifier( void )
{
   // default initialisation
   SetMethodName( "BayesClassifier" );
   SetMethodType( TMVA::Types::kBayesClassifier );
   SetTestvarName();
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
   MethodBase::ProcessOptions();
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

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
}

//_______________________________________________________________________
void  TMVA::MethodBayesClassifier::WriteWeightsToStream( ostream & o ) const
{  
   // write the weight from the training to a file (stream)
   o << "whatever" << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodBayesClassifier::ReadWeightsFromStream( istream & istr )
{
   // read back the training results from a file (stream)
   if (istr.eof());
}

//_______________________________________________________________________
Double_t TMVA::MethodBayesClassifier::GetMvaValue()
{
   // returns MVA value for given event
   Double_t myMVA = 0;

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
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "<None>" << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "<None>" << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "<None>" << Endl;
}
