// @(#)root/tmva $Id: MethodSVM.cxx,v 1.16 2006/11/02 15:44:50 andreas.hoecker Exp $    
// Author: Marcin .... 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSVM                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin ..                                                                 *
 *      + student                                                                 *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Friedman's SVM method -- not yet implemented -- dummy Class --   
//_______________________________________________________________________

#include "TMVA/MethodSVM.h"
#include "Riostream.h"

ClassImp(TMVA::MethodSVM)
   ; 

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   InitSVM();
}

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( DataSet& theData, 
                            TString theWeightFile,  
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor from weight file
   InitSVM();
}

//_______________________________________________________________________
void TMVA::MethodSVM::InitSVM( void )
{
   // default initialisation
   SetMethodName( "SVM" );
   SetMethodType( TMVA::Types::SVM );
   SetTestvarName();
}

void TMVA::MethodSVM::DeclareOptions() 
{}

void TMVA::MethodSVM::ProcessOptions() 
{
   MethodBase::ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodSVM::~MethodSVM( void )
{
   // destructor
}

//_______________________________________________________________________
void TMVA::MethodSVM::Train( void )
{
   // some training 

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
}

//_______________________________________________________________________
void  TMVA::MethodSVM::WriteWeightsToStream( ostream & o ) const
{  
   o << "whatever" << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodSVM::ReadWeightsFromStream( istream & istr )
{
   if (istr.eof());
}

//_______________________________________________________________________
Double_t TMVA::MethodSVM::GetMvaValue()
{
   // returns MVA value for given event
   Double_t myMVA = 0;

   return myMVA;
}

