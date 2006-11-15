// @(#)root/tmva $Id: MethodBayesClassifier.cxx,v 1.2 2006/11/02 15:44:50 andreas.hoecker Exp $    
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
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany                                                *
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
   ; 

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
   SetMethodType( TMVA::Types::BayesClassifier );
   SetTestvarName();
}

void TMVA::MethodBayesClassifier::DeclareOptions() 
{}

void TMVA::MethodBayesClassifier::ProcessOptions() 
{
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
   o << "whatever" << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodBayesClassifier::ReadWeightsFromStream( istream & istr )
{
   if (istr.eof());
}

//_______________________________________________________________________
Double_t TMVA::MethodBayesClassifier::GetMvaValue()
{
   // returns MVA value for given event
   Double_t myMVA = 0;

   return myMVA;
}

