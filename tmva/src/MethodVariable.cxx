// @(#)root/tmva $Id: MethodVariable.cxx,v 1.21 2006/11/02 15:44:50 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodVariable                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
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
// Wrapper class for a single variable "MVA"; this is required for      
// the evaluation of the single variable discrimination performance     
//_______________________________________________________________________

#include "TMVA/MethodVariable.h"
#include <algorithm>

ClassImp(TMVA::MethodVariable)
   ;
 
//_______________________________________________________________________
TMVA::MethodVariable::MethodVariable( TString jobName, TString methodTitle, DataSet& theData, 
                                      TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   // option string contains variable name - but not only ! 
   // there is a "Var_" prefix, which is useful in the context of later root plotting
   // so, remove this part

   SetMethodName( "Variable" );
   SetMethodType( TMVA::Types::Variable );
   SetTestvarPrefix( "" );
   SetTestvarName();

   fLogger << kVERBOSE << "uses as discriminating variable just "
           << GetOptions() << " as specified in the option" << Endl;
  
   if (0 == Data().GetTrainingTree()->FindBranch(GetOptions())) {
      Data().GetTrainingTree()->Print();
      fLogger << kFATAL << "variable " << GetOptions() << " not found in tree" << Endl;
   }
   else {
      SetMethodName ( GetMethodName() + (TString)"_" + GetOptions() );
      SetTestvarName( GetOptions() );
      fLogger << kVERBOSE << "sucessfully initialized variable as " << GetMethodName() << Endl;
   }
}

//_______________________________________________________________________
TMVA::MethodVariable::~MethodVariable( void )
{
   // destructor
}

//_______________________________________________________________________
void TMVA::MethodVariable::Train( void )
{
   // no training required

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
}

//_______________________________________________________________________
Double_t TMVA::MethodVariable::GetMvaValue()
{
   // "MVA" value is variable value
   return Data().Event().GetVal(0);
}

//_______________________________________________________________________
void  TMVA::MethodVariable::WriteWeightsToStream( ostream & o ) const
{  
   o << "";
}
  
//_______________________________________________________________________
void  TMVA::MethodVariable::ReadWeightsFromStream( istream & istr )
{
   if (istr.eof());
}
